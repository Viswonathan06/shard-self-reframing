# utils/model_client.py
import os
import logging
from typing import Optional

from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)

# Check for vLLM availability (catch ImportError and numba/NumPy compatibility errors)
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except Exception as e:
    VLLM_AVAILABLE = False
    if "numba" in str(e).lower() or "numpy" in str(type(e).__name__).lower():
        logging.debug(
            "vLLM unavailable (numba/NumPy): %s. Use HuggingFace with --use-hf, or install numba>=0.61 for NumPy 2.x.",
            e,
        )

class ContentFilteredError(Exception):
    pass

AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY        = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_MODEL      = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Lazy initialization: only create client when needed (not when using local models)
_client = None
_vllm_llm = None
_vllm_tokenizer = None
_vllm_model_name = None
# HuggingFace pipeline cache (load once per process; avoids reload on CPU after CUDA error)
_hf_pipe = None
_hf_tokenizer = None
_hf_model_name = None
_rope_ignore_keys_patch_applied = False


# Safeshift reads these for vLLM.LLM(...) kwargs. vLLM 0.19+ warns on unknown VLLM_* env vars; pop
# around LLM() only (values are already folded into llm_kw), then restore for the rest of the process.
_VLLM_SS_ENV_KEYS = (
    "VLLM_TENSOR_PARALLEL_SIZE",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_NO_GPU_UTIL_CLAMP",
    "VLLM_ENFORCE_EAGER",
    "VLLM_DISABLE_CHUNKED_PREFILL",
    "VLLM_TRUST_REMOTE_CODE",
    "VLLM_ONLY",
    "VLLM_USE_CHAT",
)


def _pop_vllm_ss_env_for_llm_ctor():
    popped = {}
    for k in _VLLM_SS_ENV_KEYS:
        if k in os.environ:
            popped[k] = os.environ.pop(k)
    return popped


def _restore_popped_env(popped: dict) -> None:
    for k, v in popped.items():
        os.environ[k] = v



def _patch_transformers_rope_check_received_keys() -> None:
    """
    Transformers 5.x: validate_rope() calls _check_received_keys with ignore_keys from the config.
    Some models (e.g. Qwen3.5) set ignore_keys_at_rope_validation as a list; the implementation does
    received_keys -= ignore_keys which requires a set, causing TypeError: set -= list.
    """
    global _rope_ignore_keys_patch_applied
    if _rope_ignore_keys_patch_applied:
        return
    try:
        from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin

        _orig = RotaryEmbeddingConfigMixin._check_received_keys

        def _check_received_keys_patched(
            rope_type, received_keys, required_keys, optional_keys=None, ignore_keys=None
        ):
            if ignore_keys is not None and not isinstance(ignore_keys, set):
                ignore_keys = set(ignore_keys)
            return _orig(rope_type, received_keys, required_keys, optional_keys, ignore_keys)

        RotaryEmbeddingConfigMixin._check_received_keys = staticmethod(_check_received_keys_patched)
        _rope_ignore_keys_patch_applied = True
    except Exception as e:
        logging.debug("transformers RoPE ignore_keys patch skipped: %s", e)


def _patch_missing_vocab_size(model) -> None:
    """Backfill config.vocab_size when the config class omits it (e.g. Qwen3_5Config).

    Newer Qwen3.5 configs store vocab size under different attribute names.
    The transformers pipeline calls config.vocab_size directly and raises AttributeError
    if it is absent, so we set it from whatever attribute is available.
    """
    try:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return
        if getattr(cfg, "vocab_size", None) is not None:
            return  # already present
        # Common alternative attribute names used by Qwen / newer configs
        for attr in ("padded_vocab_size", "hidden_size_per_partition",
                     "num_embeddings", "embed_dim"):
            v = getattr(cfg, attr, None)
            if v is not None:
                cfg.vocab_size = int(v)
                return
        # Last resort: read from the model's embedding layer
        emb = (getattr(model, "embed_tokens", None)
               or getattr(model, "wte", None)
               or getattr(getattr(model, "model", None), "embed_tokens", None))
        if emb is not None and hasattr(emb, "num_embeddings"):
            cfg.vocab_size = int(emb.num_embeddings)
    except Exception as e:
        logging.debug("_patch_missing_vocab_size skipped: %s", e)


def _sanitize_hf_model_generation_config(model, max_new_tokens_hint: int = 2048) -> None:
    """Fix hub defaults like generation_config.max_length=20 that break text-generation."""
    try:
        cfg = getattr(model, "config", None)
        mpe = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "model_max_length", None) or 8192
        try:
            mpe = int(mpe)
        except (TypeError, ValueError):
            mpe = 8192
        gc = getattr(model, "generation_config", None)
        if gc is None:
            return
        ml = getattr(gc, "max_length", None)
        need = max(int(max_new_tokens_hint) + 256, 512)
        if ml is not None and int(ml) < need:
            gc.max_length = mpe
    except Exception as e:
        logging.debug("HF generation_config sanitize skipped: %s", e)


def _is_local_model_path(name: str) -> bool:
    """True if name is an absolute path to an existing directory (local model dir)."""
    if not name or not os.path.isabs(str(name)):
        return False
    return os.path.isdir(name)


def _apply_chat_template_prompt(tokenizer, messages: list, model_name: str) -> Optional[str]:
    """
    Render chat messages to a single prompt string.
    Qwen3.x (incl. Qwen3.5) chat templates default to thinking mode; without
    enable_thinking=False, Transformers/vLLM can yield empty assistant text and P2 parses as null.
    Gemma models need special handling for chat templates.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return None
    kw = dict(tokenize=False, add_generation_prompt=True)
    mn = str(model_name).lower()
    try:
        if "qwen3" in mn and "qwen2" not in mn:
            try:
                return tokenizer.apply_chat_template(messages, **kw, enable_thinking=False)
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages, **kw, chat_template_kwargs={"enable_thinking": False}
                )
        elif "gemma-4" in mn or "gemma4" in mn:
            # Gemma 4 uses specific chat template format
            try:
                return tokenizer.apply_chat_template(messages, **kw)
            except Exception:
                # Fallback to manual Gemma format
                prompt_parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt_parts.append(f"{content}")
                    elif role == "user":
                        prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                    elif role == "assistant":
                        prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
                prompt_parts.append("<start_of_turn>model\n")
                return "\n".join(prompt_parts)
        return tokenizer.apply_chat_template(messages, **kw)
    except Exception:
        return None


def _vllm_chat_extra_kwargs(model_name: str) -> dict:
    """Extra keyword args for vLLM LLM.chat (e.g. Qwen3 thinking mode, Gemma chat templates)."""
    mn = str(model_name).lower()
    if "qwen3" in mn and "qwen2" not in mn:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    elif "gemma-4" in mn or "gemma4" in mn:
        # Gemma 4 specific settings for optimal performance
        return {"chat_template_kwargs": {"add_generation_prompt": True}}
    return {}


def _vllm_completion_text(request_output, tokenizer) -> str:
    """Decode first completion; fall back to token_ids if .text is empty."""
    if not request_output.outputs:
        return ""
    comp = request_output.outputs[0]
    t = (comp.text or "").strip()
    if t:
        return t
    tok = getattr(comp, "token_ids", None)
    if tok and tokenizer is not None:
        try:
            decoded = tokenizer.decode(list(tok), skip_special_tokens=True)
            if decoded.strip():
                return decoded.strip()
        except Exception:
            pass
    return ""


def _vllm_infer_chat_or_generate(llm, tokenizer, messages: list, model_name: str, sampling_params) -> str:
    """
    Prefer LLM.chat() so vLLM runs the model-specific chat preprocessor (e.g. Gemma3).
    Falls back to apply_chat_template + generate when VLLM_USE_CHAT=0 or chat is missing.
    """
    use_chat = os.environ.get("VLLM_USE_CHAT", "1").strip().lower() in ("1", "true", "yes")
    extra = _vllm_chat_extra_kwargs(model_name)
    if use_chat and hasattr(llm, "chat"):
        outputs = llm.chat(
            messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            **extra,
        )
        return _vllm_completion_text(outputs[0], tokenizer)

    prompt = _apply_chat_template_prompt(tokenizer, messages, model_name)
    if prompt is None:
        sys_p, usr_p = "", ""
        for m in messages:
            if m.get("role") == "system":
                sys_p = m.get("content") or ""
            elif m.get("role") == "user":
                usr_p = m.get("content") or ""
        prompt = f"System: {sys_p}\n\nUser: {usr_p}\n\nAssistant:"
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    return _vllm_completion_text(outputs[0], tokenizer)


def _get_azure_client():
    """Lazy initialization of Azure OpenAI client. Only called when actually needed."""
    global _client
    if _client is not None:
        return _client
    
    # Check if local model is enabled - if so, we don't need Azure client
    USE_LOCAL_MODEL_ENV = os.getenv("USE_LOCAL_MODEL", "false").strip().lower()
    USE_LOCAL_MODEL = USE_LOCAL_MODEL_ENV == "true" or USE_LOCAL_MODEL_ENV == "1"
    
    if USE_LOCAL_MODEL:
        # Don't initialize Azure client if using local model
        return None
    
    # Validate credentials only when actually needed
    if not AZURE_ENDPOINT:
        raise ValueError(
            "❌ Missing AZURE_OPENAI_ENDPOINT environment variable.\n"
            "   Please set it in your .env file or environment:\n"
            "   AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com/"
        )

    if not AZURE_KEY:
        raise ValueError(
            "❌ Missing AZURE_OPENAI_API_KEY environment variable.\n"
            "   Please set it in your .env file or environment:\n"
            "   AZURE_OPENAI_API_KEY=your_api_key_here"
        )

    # Note: Azure API URL logging is handled in evaluate_p2_selection.py to log once at startup

    # Normalize endpoint format
    # Azure OpenAI can use either:
    # 1. OpenAI endpoint: https://YOUR_RESOURCE.openai.azure.com/
    # 2. Cognitive Services endpoint: https://YOUR_RESOURCE.cognitiveservices.azure.com/
    endpoint_normalized = AZURE_ENDPOINT.strip().rstrip('/')

    # Convert Cognitive Services endpoint to OpenAI format if needed
    if 'cognitiveservices.azure.com' in endpoint_normalized:
        # Extract resource name and convert to OpenAI endpoint format
        # e.g., https://resource.cognitiveservices.azure.com/ -> https://resource.openai.azure.com/
        endpoint_normalized = endpoint_normalized.replace('cognitiveservices.azure.com', 'openai.azure.com')
        logging.info(f"Converted Cognitive Services endpoint to OpenAI format")

    try:
        _client = AzureOpenAI(
            api_version=AZURE_API_VER,
            azure_endpoint=endpoint_normalized,
            api_key=AZURE_KEY,
        )
        return _client
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "invalid subscription key" in error_msg.lower() or "Access denied" in error_msg:
            # Check if local model is available as alternative
            USE_LOCAL_MODEL_ENV = os.getenv("USE_LOCAL_MODEL", "false").strip().lower()
            
            # Check if API key looks valid (should be ~32 chars, alphanumeric)
            key_preview = AZURE_KEY[:8] + "..." + AZURE_KEY[-4:] if AZURE_KEY and len(AZURE_KEY) > 12 else "INVALID"
            
            error_suggestion = (
                f"❌ Azure OpenAI Authentication Failed (401)\n"
                f"\n"
                f"   Troubleshooting steps:\n"
                f"   1. Verify API key is correct:\n"
                f"      - Key preview: {key_preview}\n"
                f"      - Get key from: Azure Portal → Your Resource → Keys and Endpoint\n"
                f"   2. Verify endpoint matches your resource:\n"
                f"      - Original: {AZURE_ENDPOINT[:70]}...\n"
                f"      - Converted: {endpoint_normalized[:70]}...\n"
                f"   3. Verify deployment exists:\n"
                f"      - Model: {AZURE_MODEL}\n"
                f"      - Check Azure Portal → Your Resource → Deployments\n"
                f"   4. Check .env file location:\n"
                f"      - project root/.env\n"
                f"\n"
                f"   Common issues:\n"
                f"   - API key expired or regenerated\n"
                f"   - Wrong resource/endpoint\n"
                f"   - Deployment name doesn't match AZURE_OPENAI_MODEL\n"
                f"   - Subscription doesn't have access to Azure OpenAI\n"
            )
            
            # Suggest using local models if not already set
            if USE_LOCAL_MODEL_ENV != "true":
                error_suggestion += (
                    f"\n"
                    f"   💡 ALTERNATIVE: Use local models instead of Azure API.\n"
                    f"      Add to your .env file:\n"
                    f"      USE_LOCAL_MODEL=true\n"
                    f"      LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct  # or your preferred model\n"
                )
            
            raise ValueError(error_suggestion) from e
        raise ValueError(
            f"❌ Failed to initialize Azure OpenAI client: {e}\n"
            f"   Endpoint: {endpoint_normalized[:60]}...\n"
            f"   Please check your credentials in .env file"
        ) from e

def get_client():
    """Get the Azure OpenAI client. Returns None if local model is enabled."""
    return _get_azure_client()

def _get_vllm_model(model_name: str, use_hf: bool = False):
    """Get or initialize vLLM model for GPU inference."""
    global _vllm_llm, _vllm_tokenizer, _vllm_model_name
    model_name = str(model_name)  # ensure string for vLLM/tokenizer (Path → SentencePiece "not a string")
    
    # Check if we should use vLLM
    if not VLLM_AVAILABLE or use_hf:
        return None, None
    
    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
    except ImportError:
        return None, None
    
    # Return cached model if same name
    if _vllm_llm is not None and _vllm_model_name == model_name:
        return _vllm_llm, _vllm_tokenizer
    
    # Tensor parallel size from env (e.g. 4 for 27B/70B across 4 GPUs)
    _tp = os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")
    try:
        tensor_parallel_size = int(_tp)
    except ValueError:
        tensor_parallel_size = 1
    if tensor_parallel_size < 1:
        tensor_parallel_size = 1

    # TP must not exceed visible CUDA devices (Slurm may grant fewer GPUs than #SBATCH --gres).
    try:
        n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_visible > 0 and tensor_parallel_size > n_visible:
            logging.warning(
                "VLLM_TENSOR_PARALLEL_SIZE=%s > torch.cuda.device_count()=%s; clamping tensor_parallel_size to %s "
                "(avoids PyTorch 'device >= num_gpus' during vLLM worker init).",
                tensor_parallel_size,
                n_visible,
                n_visible,
            )
            tensor_parallel_size = n_visible
    except Exception as e:
        logging.debug("TP vs device_count clamp skipped: %s", e)

    # Max context (KV cache scales with this). Lower if TP workers OOM (e.g. 2048 for Qwen3.5 on 48GB).
    _mlm = os.environ.get("VLLM_MAX_MODEL_LEN", "4096")
    try:
        max_model_len = int(_mlm)
    except ValueError:
        max_model_len = 4096
    if max_model_len < 512:
        max_model_len = 512

    # Initialize vLLM model
    # Get GPU memory utilization from env (default 0.85 to leave some headroom)
    # But check available memory and adjust if needed
    gpu_memory_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
    
    # Check available GPU memory and adjust utilization if needed
    no_clamp = os.environ.get("VLLM_NO_GPU_UTIL_CLAMP", "").strip().lower() in ("1", "true", "yes")
    if no_clamp:
        logging.info(
            "VLLM_NO_GPU_UTIL_CLAMP=1: skipping GPU utilization clamp (parent CUDA usage would otherwise "
            "reduce utilization and can starve KV cache at fixed max_model_len)."
        )
    try:
        import torch
        if not no_clamp and torch.cuda.is_available():
            min_free_gb = float('inf')
            total_gb = None
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / (1024**3)
                # Get free memory (this is approximate, actual free memory may differ)
                # We'll use a conservative estimate
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                free_gb = total_gb - allocated
                min_free_gb = min(min_free_gb, free_gb)
            
            # If free memory is less than what we'd need with current utilization, reduce it
            if min_free_gb > 0 and total_gb:
                # Calculate what utilization would fit in available memory (with 2GB safety margin)
                safe_util = max(0.1, min(gpu_memory_util, (min_free_gb - 2) / total_gb))
                if safe_util < gpu_memory_util:
                    logging.warning(f"Reducing GPU memory utilization from {gpu_memory_util} to {safe_util:.2f} to fit available memory ({min_free_gb:.2f} GB free per GPU)")
                    gpu_memory_util = safe_util
    except Exception as e:
        logging.warning(f"Could not check GPU memory, using default utilization: {e}")
    
    logging.info("="*80)
    logging.info("USING vLLM FOR INFERENCE (2-10x faster for batch processing)")
    logging.info("="*80)
    logging.info(f"Model: {model_name}")
    logging.info(f"Max model length: {max_model_len}")
    logging.info(f"Dtype: bfloat16")
    logging.info(f"Tensor parallel size: {tensor_parallel_size}")
    logging.info(f"GPU memory utilization: {gpu_memory_util}")
    
    _patch_transformers_rope_check_received_keys()
    
    try:
        llm_kw = dict(
            model=model_name,
            max_model_len=max_model_len,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_util,
        )
        # vLLM warns on >2 PCIe GPUs; explicit True avoids custom AR path issues on many nodes.
        if tensor_parallel_size > 1:
            llm_kw["disable_custom_all_reduce"] = True
        if os.environ.get("VLLM_ENFORCE_EAGER", "").strip().lower() in ("1", "true", "yes"):
            llm_kw["enforce_eager"] = True
        if os.environ.get("VLLM_DISABLE_CHUNKED_PREFILL", "").strip().lower() in ("1", "true", "yes"):
            llm_kw["enable_chunked_prefill"] = False
        # Qwen3.x and Gemma models often need remote code; also set VLLM_TRUST_REMOTE_CODE=1 to force.
        _trc = os.environ.get("VLLM_TRUST_REMOTE_CODE", "").strip().lower() in ("1", "true", "yes")
        if not _trc:
            _mn = str(model_name).lower()
            if "qwen3" in _mn and "qwen2" not in _mn:
                _trc = True
            elif "gemma-3" in _mn or "gemma3" in _mn:
                _trc = True
            elif "gemma-4" in _mn or "gemma4" in _mn:
                _trc = True
        llm_kw["trust_remote_code"] = _trc
        _popped_vllm_env = _pop_vllm_ss_env_for_llm_ctor()
        try:
            try:
                _vllm_llm = LLM(**llm_kw)
            except TypeError as te:
                if "enable_chunked_prefill" in llm_kw:
                    llm_kw.pop("enable_chunked_prefill", None)
                    logging.warning("Retrying vLLM without enable_chunked_prefill (unsupported on this vLLM): %s", te)
                    _vllm_llm = LLM(**llm_kw)
                else:
                    raise
        finally:
            _restore_popped_env(_popped_vllm_env)
        # Prefer vLLM's tokenizer: it already loaded with the engine (correct transformers/hub pairing).
        # A second AutoTokenizer.from_pretrained here used to fail on Qwen3.5 + transformers<5.2 (qwen3_5),
        # which aborted the whole try and looked like "vLLM not working" even though LLM() had started.
        try:
            _vllm_tokenizer = _vllm_llm.get_tokenizer()
        except Exception as tok_e:
            if os.environ.get("VLLM_ONLY", "").strip().lower() in ("1", "true", "yes"):
                raise RuntimeError(
                    f"vLLM get_tokenizer() failed and VLLM_ONLY=1 (no HuggingFace tokenizer fallback): {tok_e}"
                ) from tok_e
            logging.warning(f"vLLM get_tokenizer() failed ({tok_e}); loading AutoTokenizer")
            tokenizer_kw = {"use_fast": True, "trust_remote_code": True}
            if _is_local_model_path(model_name):
                tokenizer_kw["local_files_only"] = True
            _vllm_tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kw)
        _vllm_model_name = model_name
        logging.info("vLLM model loaded successfully")
        return _vllm_llm, _vllm_tokenizer
    except Exception as e:
        logging.warning(f"Failed to load vLLM model: {e}")
        if os.environ.get("VLLM_ONLY", "").strip().lower() in ("1", "true", "yes"):
            raise RuntimeError(
                f"vLLM failed to load and VLLM_ONLY=1 (HuggingFace fallback disabled): {e}"
            ) from e
        logging.warning("Falling back to HuggingFace")
        return None, None

def batch_call_model(
    prompts: list,
    max_tokens: int = 512,
    temperature: float = 0.7,
    use_hf: bool = False,
    model_name: str = None,
) -> list:
    """Batch inference via vLLM: send all (system, user) pairs in one llm.chat() call.

    Falls back to sequential call_model() when vLLM is unavailable (Azure/HF path).

    Args:
        prompts: list of (system_prompt, user_prompt) tuples
        Returns: list of str responses, same length as prompts (None on error).
    """
    if not prompts:
        return []

    load_dotenv()
    USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").strip().lower() in ("true", "1")
    LOCAL_MODEL_NAME = str(model_name or os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct"))

    if USE_LOCAL_MODEL:
        llm, tokenizer = _get_vllm_model(LOCAL_MODEL_NAME, use_hf=use_hf)
        if llm is not None and tokenizer is not None:
            try:
                conversations = []
                for sys_p, usr_p in prompts:
                    msgs = []
                    if sys_p:
                        msgs.append({"role": "system", "content": sys_p})
                    msgs.append({"role": "user", "content": usr_p})
                    conversations.append(msgs)

                stop_tokens = []
                if "gemma" in LOCAL_MODEL_NAME.lower():
                    stop_tokens = ["<end_of_turn>", "<eos>", "</s>"]

                sampling_params = SamplingParams(
                    temperature=max(0.1, temperature),
                    max_tokens=max_tokens,
                    stop=stop_tokens if stop_tokens else None,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )

                use_chat = os.environ.get("VLLM_USE_CHAT", "1").strip().lower() in ("1", "true", "yes")
                extra = _vllm_chat_extra_kwargs(LOCAL_MODEL_NAME)
                if use_chat and hasattr(llm, "chat"):
                    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False, **extra)
                    return [_vllm_completion_text(o, tokenizer) for o in outputs]

                # Fallback: apply chat template per prompt, batch generate
                rendered = [
                    _apply_chat_template_prompt(tokenizer, msgs, LOCAL_MODEL_NAME) or ""
                    for msgs in conversations
                ]
                outputs = llm.generate(rendered, sampling_params=sampling_params)
                return [_vllm_completion_text(o, tokenizer) for o in outputs]

            except Exception as e:
                if os.environ.get("VLLM_ONLY", "").strip().lower() in ("1", "true", "yes"):
                    raise RuntimeError(f"vLLM batch generation failed and VLLM_ONLY=1: {e}") from e
                logging.warning("vLLM batch failed (%s); falling back to sequential call_model", e)

    # Sequential fallback (Azure or HF)
    results = []
    for sys_p, usr_p in prompts:
        try:
            results.append(call_model(sys_p, usr_p, max_tokens=max_tokens,
                                      temperature=temperature, use_hf=use_hf, model_name=model_name))
        except Exception as e:
            logging.error("batch_call_model sequential fallback failed: %s", e)
            results.append(None)
    return results


def _parse_thinking_output(raw: str):
    """Split a Qwen3 thinking-mode output into (thinking, response).

    Qwen3 thinking outputs look like:
        <think>\n...reasoning...\n</think>\n\nFinal response text
    Returns (thinking_text, response_text). If no <think> tag, thinking is "".
    """
    import re
    if not raw:
        return "", ""
    # Full tags present: <think>...</think>\n\nResponse
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        response = raw[m.end():].strip()
        return thinking, response
    # Qwen3 chat template injects <think> as a prefix token that vLLM strips,
    # so the returned text is: [thinking]\n</think>\n\nResponse
    m = re.search(r"(.*?)</think>", raw, re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        response = raw[m.end():].strip()
        return thinking, response
    return "", raw.strip()


def batch_call_model_with_thinking(
    prompts: list,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    use_hf: bool = False,
    model_name: str = None,
) -> list:
    """Batch inference with Qwen3 thinking mode enabled.

    Returns a list of (thinking: str, response: str) tuples — one per prompt.
    The thinking field contains the raw <think>...</think> content.
    Falls back to batch_call_model (thinking disabled) for non-Qwen3 models.
    """
    if not prompts:
        return []
    load_dotenv()
    USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").strip().lower() in ("true", "1")
    LOCAL_MODEL_NAME = str(model_name or os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct"))

    mn = LOCAL_MODEL_NAME.lower()
    is_qwen3 = "qwen3" in mn and "qwen2" not in mn

    if USE_LOCAL_MODEL:
        llm, tokenizer = _get_vllm_model(LOCAL_MODEL_NAME, use_hf=use_hf)
        if llm is not None and tokenizer is not None:
            try:
                conversations = []
                for sys_p, usr_p in prompts:
                    msgs = []
                    if sys_p:
                        msgs.append({"role": "system", "content": sys_p})
                    msgs.append({"role": "user", "content": usr_p})
                    conversations.append(msgs)

                sampling_params = SamplingParams(
                    temperature=max(0.1, temperature),
                    max_tokens=max_tokens,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )

                use_chat = os.environ.get("VLLM_USE_CHAT", "1").strip().lower() in ("1", "true", "yes")
                if is_qwen3:
                    extra = {"chat_template_kwargs": {"enable_thinking": True}}
                else:
                    extra = _vllm_chat_extra_kwargs(LOCAL_MODEL_NAME)

                if use_chat and hasattr(llm, "chat"):
                    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False, **extra)
                    raws = [_vllm_completion_text(o, tokenizer) for o in outputs]
                else:
                    if is_qwen3:
                        rendered = [
                            _apply_chat_template_prompt_with_thinking(tokenizer, msgs, LOCAL_MODEL_NAME) or ""
                            for msgs in conversations
                        ]
                    else:
                        rendered = [
                            _apply_chat_template_prompt(tokenizer, msgs, LOCAL_MODEL_NAME) or ""
                            for msgs in conversations
                        ]
                    outputs = llm.generate(rendered, sampling_params=sampling_params)
                    raws = [_vllm_completion_text(o, tokenizer) for o in outputs]

                return [_parse_thinking_output(r) for r in raws]

            except Exception as e:
                if os.environ.get("VLLM_ONLY", "").strip().lower() in ("1", "true", "yes"):
                    raise RuntimeError(f"vLLM batch_with_thinking failed and VLLM_ONLY=1: {e}") from e
                logging.warning("vLLM batch_with_thinking failed (%s); falling back to sequential", e)

    # Sequential fallback — thinking not available, return ("", response) tuples
    results = []
    for sys_p, usr_p in prompts:
        try:
            resp = call_model(sys_p, usr_p, max_tokens=max_tokens,
                              temperature=temperature, use_hf=use_hf, model_name=model_name)
            results.append(("", resp or ""))
        except Exception as e:
            logging.error("batch_call_model_with_thinking sequential fallback failed: %s", e)
            results.append(("", ""))
    return results


def _apply_chat_template_prompt_with_thinking(tokenizer, messages: list, model_name: str) -> Optional[str]:
    """Like _apply_chat_template_prompt but with enable_thinking=True for Qwen3."""
    if not hasattr(tokenizer, "apply_chat_template"):
        return None
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:
        try:
            return tokenizer.apply_chat_template(messages, **kw, enable_thinking=True)
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, **kw, chat_template_kwargs={"enable_thinking": True}
            )
    except Exception:
        return None


def call_model(system_prompt: str,
               user_prompt: str,
               max_tokens: int = 512,
               temperature: float = 0.7,
               enforce_json: bool = False,
               use_hf: bool = False,
               model_name: str = None) -> str:
    """
    Wrapper for Azure OpenAI chat completion API or local model.
    Uses local model if USE_LOCAL_MODEL=true, otherwise uses Azure API.

    - use_hf: if True and using local model, use HuggingFace backend only (skip vLLM).
    - model_name: if set and using local model, use this HuggingFace model id instead of LOCAL_MODEL env.
    - If enforce_json=True, we ask the model to return a JSON object and enable JSON mode if supported.
    - Raises ContentFilteredError if completion content is empty (often content filter).
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    USE_LOCAL_MODEL_ENV = os.getenv("USE_LOCAL_MODEL", "false").strip().lower()
    USE_LOCAL_MODEL = USE_LOCAL_MODEL_ENV == "true" or USE_LOCAL_MODEL_ENV == "1"
    LOCAL_MODEL_NAME = model_name or os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    LOCAL_MODEL_NAME = str(LOCAL_MODEL_NAME)  # ensure string (avoids SentencePiece "not a string" with Path)
    
    # Ensure LOCAL_MODEL_NAME is not None
    if not LOCAL_MODEL_NAME:
        raise ValueError(
            "LOCAL_MODEL_NAME is None. "
            "Either pass model_name parameter or set LOCAL_MODEL environment variable."
        )
    
    # Use local model if enabled
    if USE_LOCAL_MODEL:
        # Try vLLM first (faster on GPU) unless use_hf=True
        llm, tokenizer = _get_vllm_model(LOCAL_MODEL_NAME, use_hf=use_hf)
        if llm is not None and tokenizer is not None:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})

                # Improve sampling params for better generation, especially for Gemma
                stop_tokens = []
                mn_lower = str(LOCAL_MODEL_NAME).lower()
                if "gemma" in mn_lower:
                    stop_tokens = ["<end_of_turn>", "<eos>", "</s>"]
                
                sampling_params = SamplingParams(
                    temperature=max(0.1, temperature),  # Ensure minimum temperature
                    max_tokens=max_tokens,
                    stop=stop_tokens if stop_tokens else None,
                    top_p=0.9,
                    repetition_penalty=1.1,  # Reduce repetition
                )
                return _vllm_infer_chat_or_generate(
                    llm, tokenizer, messages, LOCAL_MODEL_NAME, sampling_params
                )
            except Exception as e:
                if os.environ.get("VLLM_ONLY", "").strip().lower() in ("1", "true", "yes"):
                    raise RuntimeError(
                        f"vLLM generation failed and VLLM_ONLY=1 (HuggingFace fallback disabled): {e}"
                    ) from e
                logging.warning(f"vLLM generation failed: {e}, falling back to HuggingFace")
        
        if os.environ.get("VLLM_ONLY", "").strip().lower() in ("1", "true", "yes"):
            raise RuntimeError(
                "vLLM is not available (llm/tokenizer is None) and VLLM_ONLY=1 — "
                "fix vLLM startup (see job logs / VLLM_TRUST_REMOTE_CODE / transformers>=5.2 with pip --no-deps)."
            )
        
        # Fall back to HuggingFace transformers directly (cache pipe so we load once per process)
        try:
            global _hf_pipe, _hf_tokenizer, _hf_model_name
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig as HFGenerationConfig
            import torch

            if _hf_pipe is not None and _hf_model_name == LOCAL_MODEL_NAME:
                tokenizer = _hf_tokenizer
                pipe = _hf_pipe
            else:
                # Load tokenizer and model (use_fast=True avoids SentencePiece "not a string" with Llama local paths)
                tokenizer_kw = {"trust_remote_code": True, "use_fast": True}
                if _is_local_model_path(LOCAL_MODEL_NAME):
                    tokenizer_kw["local_files_only"] = True  # avoid HFValidationError when path is a local dir
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, **tokenizer_kw)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model (use device_map="auto" so 70B+ spreads across all GPUs; needs accelerate)
                # Use float16 (not bfloat16) so sampling does not trigger "probability tensor contains inf/nan" CUDA assert
                device = "cuda" if torch.cuda.is_available() else "cpu"
                n_gpu = torch.cuda.device_count() if device == "cuda" else 0
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                }
                if _is_local_model_path(LOCAL_MODEL_NAME):
                    model_kwargs["local_files_only"] = True
                try:
                    import accelerate
                    if device == "cuda":
                        model_kwargs["device_map"] = "auto"  # use all visible GPUs (required for 70B on 4x A6000)
                except ImportError:
                    if n_gpu > 1:
                        logging.warning("accelerate not installed; model will load on one GPU only (may OOM for 70B). Install: pip install accelerate")
                    model_kwargs["device_map"] = None
                model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, **model_kwargs)
                # Patch missing vocab_size on newer architectures (e.g. Qwen3_5Config) where
                # the attribute is defined in config.json but the config class doesn't expose it.
                _patch_missing_vocab_size(model)
                _sanitize_hf_model_generation_config(model, max_new_tokens_hint=max_tokens)
                if model_kwargs.get("device_map") != "auto" and device == "cuda":
                    model = model.to("cuda:0")
                if device == "cuda" and model_kwargs.get("device_map") == "auto":
                    dmap = getattr(model, "hf_device_map", None) or getattr(model, "device_map", None) or {}
                    num_devices = len(set(dmap.values())) if dmap else n_gpu
                    if num_devices >= 2:
                        print(f"✓ Model loaded successfully across {num_devices} GPUs (device_map=auto)")
                    else:
                        print(f"✓ Model loaded on GPU (device_map=auto, {n_gpu} GPU(s) visible)")
                # Pipeline: omit device when model uses device_map=auto (pipeline then uses first device in map for I/O)
                # Use float16 on GPU to avoid bfloat16 sampling assert ("probability tensor contains inf/nan")
                pipe_dtype = torch.float16 if device == "cuda" else torch.float32
                pipe_kw = dict(
                    model=model,
                    tokenizer=tokenizer,
                    dtype=pipe_dtype,
                )
                if model_kwargs.get("device_map") != "auto":
                    pipe_kw["device"] = 0 if device == "cuda" else -1
                pipe = pipeline("text-generation", **pipe_kw)
                _hf_pipe = pipe
                _hf_tokenizer = tokenizer
                _hf_model_name = LOCAL_MODEL_NAME
            
            # Format prompt
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            prompt = _apply_chat_template_prompt(tokenizer, messages, LOCAL_MODEL_NAME)
            if prompt is None:
                prompt = f"{system_prompt}\n\n{user_prompt}\n\nResponse:"
            
            # Ensure pad/eos are Python ints (avoids CUDA device-side assert from tensor or wrong type)
            _pad = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
            _eos = getattr(tokenizer, "eos_token_id", None)
            if _pad is not None:
                _pad = int(_pad)
            if _eos is not None:
                _eos = int(_eos)
            if _pad is None and _eos is not None:
                _pad = _eos
            # Truncate prompt to model max length to avoid device-side assert (e.g. LLaMA 4096)
            model = pipe.model
            max_len = getattr(getattr(model, "config", None), "model_max_length", None) or getattr(getattr(model, "config", None), "max_position_embeddings", 4096)
            if isinstance(max_len, int) and max_len > 0:
                enc = tokenizer.encode(prompt, add_special_tokens=False)
                if len(enc) > max_len - max_tokens - 16:
                    prompt = tokenizer.decode(enc[: max_len - max_tokens - 16], skip_special_tokens=False)
            # Use greedy decoding when temperature is low to avoid "probability tensor contains inf/nan" CUDA assert in sampling
            t = float(temperature)
            use_sample = t > 0.5
            _pid = _pad if _pad is not None else _eos
            _eid = _eos if _eos is not None else _pad
            if _pid is None:
                _pid = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
            if _eid is None:
                _eid = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None

            # Single GenerationConfig avoids merging hub defaults (max_length=20) with call kwargs.
            gen_cfg = HFGenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=use_sample,
                pad_token_id=_pid,
                eos_token_id=_eid,
            )
            if use_sample:
                gen_cfg.temperature = max(0.01, t)
            outputs = pipe(prompt, generation_config=gen_cfg, return_full_text=False)

            result = outputs[0]["generated_text"].strip()
            return result
            
        except ImportError as e:
            error_msg = (
                f"❌ Local model not available: {e}\n"
                f"   Install required packages: pip install transformers torch accelerate"
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"❌ Local model failed: {e}\n"
                f"   Model: {LOCAL_MODEL_NAME}\n"
                f"   Make sure the model path is correct and all dependencies are installed."
            )
            logging.error(error_msg)
            import traceback
            logging.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e

    # If USE_LOCAL_MODEL is not set, use Azure API
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    kwargs = dict(
        model=AZURE_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # Enable JSON mode if available in your SDK/API version.
    if enforce_json:
        kwargs["response_format"] = {"type": "json_object"}

    # Get Azure client (lazy initialization)
    try:
        azure_client = _get_azure_client()
        if azure_client is None:
            raise ValueError(
                "Azure client not available. "
                "Set USE_LOCAL_MODEL=true in .env to use local models, "
                "or fix your Azure credentials."
            )
    except ValueError as e:
        # If it's a 401 error with helpful message, re-raise it
        if "401" in str(e) or "Authentication Failed" in str(e):
            raise
        # Otherwise, provide helpful error
        raise ValueError(
            f"❌ Cannot use Azure API: {e}\n"
            f"   💡 To use local models instead, add to your .env file:\n"
            f"      USE_LOCAL_MODEL=true\n"
            f"      LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct"
        ) from e
    
    try:
        completion = azure_client.chat.completions.create(**kwargs)
    except Exception as e:
        msg = str(e)
        # Check for 401 authentication errors
        if "401" in msg or "invalid subscription key" in msg.lower() or "Access denied" in msg:
            USE_LOCAL_MODEL_ENV = os.getenv("USE_LOCAL_MODEL", "false").strip().lower()
            
            error_msg = (
                f"❌ Azure OpenAI API call failed with 401 Authentication Error\n"
                f"   Error: {msg}\n"
            )
            if USE_LOCAL_MODEL_ENV != "true":
                error_msg += (
                    f"\n"
                    f"   💡 ALTERNATIVE: Use local models instead of Azure API.\n"
                    f"      Add to your .env file:\n"
                    f"      USE_LOCAL_MODEL=true\n"
                    f"      LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct  # or your preferred model\n"
                )
            
            raise ValueError(error_msg) from e
        
        if ("content_filter" in msg
            or "ResponsibleAIPolicyViolation" in msg
            or 'status": 400' in msg):
            raise ContentFilteredError(msg)
        raise

    content = (completion.choices[0].message.content or "").strip()

    # Azure can "succeed" with empty content if filtered; treat as filtered.
    if not content:
        raise ContentFilteredError("Empty/blocked completion (likely content filter).")

    return content


def call_model_batch(system_prompt: str,
                     user_prompts: list,
                     max_tokens: int = 512,
                     temperature: float = 0.7,
                     enforce_json: bool = False,
                     max_parallel: int = 10) -> list:
    """
    Batch version of call_model. Processes multiple user prompts in parallel.
    Uses local model if USE_LOCAL_MODEL=true, otherwise uses Azure API.
    
    Args:
        system_prompt: System prompt (same for all)
        user_prompts: List of user prompts to process
        max_tokens: Max tokens per completion
        temperature: Temperature setting
        enforce_json: Whether to enforce JSON output
        max_parallel: Maximum parallel requests (for API) or batch size (for local)
    
    Returns:
        List of responses (same order as user_prompts)
        Responses that failed will be None in the list
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    USE_LOCAL_MODEL_ENV = os.getenv("USE_LOCAL_MODEL", "false").strip().lower()
    USE_LOCAL_MODEL = USE_LOCAL_MODEL_ENV == "true" or USE_LOCAL_MODEL_ENV == "1"
    LOCAL_MODEL_NAME = str(os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    
    # Use local model if enabled
    if USE_LOCAL_MODEL:
        print(f"🔧 Attempting to use local model for P2 generation: {LOCAL_MODEL_NAME}")
        
        # Try vLLM first (faster on GPU, especially for batches)
        llm, tokenizer = _get_vllm_model(LOCAL_MODEL_NAME)
        if llm is not None and tokenizer is not None:
            try:
                print(f"✓ Using vLLM for batch inference: {LOCAL_MODEL_NAME}")

                convs = []
                for user_prompt in user_prompts:
                    m = []
                    if system_prompt:
                        m.append({"role": "system", "content": system_prompt})
                    m.append({"role": "user", "content": user_prompt})
                    convs.append(m)

                print(f"  Processing {len(convs)} prompts in batch with vLLM...")

                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=None,
                )
                use_chat = os.environ.get("VLLM_USE_CHAT", "1").strip().lower() in ("1", "true", "yes")
                extra = _vllm_chat_extra_kwargs(LOCAL_MODEL_NAME)
                if use_chat and hasattr(llm, "chat"):
                    outputs = llm.chat(
                        convs,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                        **extra,
                    )
                else:
                    prompts = []
                    for conv in convs:
                        p = _apply_chat_template_prompt(tokenizer, conv, LOCAL_MODEL_NAME)
                        if p is None:
                            up = conv[-1].get("content", "") if conv else ""
                            p = f"System: {system_prompt}\n\nUser: {up}\n\nAssistant:"
                        prompts.append(p)
                    outputs = llm.generate(prompts, sampling_params=sampling_params)

                formatted_results = []
                for output in outputs:
                    result = _vllm_completion_text(output, tokenizer)
                    if result and result != "{}":
                        formatted_results.append(result)
                    else:
                        formatted_results.append(None)
                
                print(f"✓ Generated {len([r for r in formatted_results if r])} responses from vLLM")
                return formatted_results
            except Exception as e:
                if os.environ.get("VLLM_ONLY", "").strip().lower() in ("1", "true", "yes"):
                    raise RuntimeError(
                        f"vLLM batch generation failed and VLLM_ONLY=1 (HuggingFace fallback disabled): {e}"
                    ) from e
                logging.warning(f"vLLM batch generation failed: {e}, falling back to HuggingFace")
                import traceback
                print(traceback.format_exc())
        
        if os.environ.get("VLLM_ONLY", "").strip().lower() in ("1", "true", "yes"):
            raise RuntimeError(
                "vLLM is not available for batch and VLLM_ONLY=1 — fix vLLM startup or set VLLM_ONLY=0."
            )
        
        # Fall back to HuggingFace transformers directly
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            # Load tokenizer and model (use_fast=True avoids SentencePiece "not a string" with Llama local paths)
            tokenizer_kw = {"trust_remote_code": True, "use_fast": True}
            if _is_local_model_path(LOCAL_MODEL_NAME):
                tokenizer_kw["local_files_only"] = True
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, **tokenizer_kw)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model (device_map="auto" spreads 70B across all GPUs; needs accelerate)
            # Use float16 (not bfloat16) to avoid "probability tensor contains inf/nan" in sampling
            device = "cuda" if torch.cuda.is_available() else "cpu"
            n_gpu = torch.cuda.device_count() if device == "cuda" else 0
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            }
            if _is_local_model_path(LOCAL_MODEL_NAME):
                model_kwargs["local_files_only"] = True
            try:
                import accelerate
                if device == "cuda":
                    model_kwargs["device_map"] = "auto"
            except ImportError:
                if n_gpu > 1:
                    logging.warning("accelerate not installed; model on one GPU only (may OOM for 70B). pip install accelerate")
                model_kwargs["device_map"] = None
            model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, **model_kwargs)
            if model_kwargs.get("device_map") != "auto" and device == "cuda":
                model = model.to("cuda:0")
            if device == "cuda" and model_kwargs.get("device_map") == "auto":
                dmap = getattr(model, "hf_device_map", None) or getattr(model, "device_map", None) or {}
                num_devices = len(set(dmap.values())) if dmap else n_gpu
                if num_devices >= 2:
                    print(f"✓ Model loaded successfully across {num_devices} GPUs (device_map=auto)")
                else:
                    print(f"✓ Model loaded on GPU (device_map=auto, {n_gpu} GPU(s) visible)")
            print(f"✓ Local model loaded successfully: {LOCAL_MODEL_NAME}")
            
            # Prepare prompts
            formatted_prompts = []
            for user_prompt in user_prompts:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                
                prompt = _apply_chat_template_prompt(tokenizer, messages, LOCAL_MODEL_NAME)
                if prompt is None:
                    prompt = f"{system_prompt}\n\n{user_prompt}\n\nResponse:"
                formatted_prompts.append(prompt)
            
            print(f"  Processing {len(formatted_prompts)} prompts in batch...")
            
            # Batch generate
            results = []
            batch_size = max_parallel * 2
            for i in range(0, len(formatted_prompts), batch_size):
                batch_prompts = formatted_prompts[i:i+batch_size]
                
                # Tokenize batch
                tokenized = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )
                
                if device == "cuda":
                    tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
                
                # Use greedy when temperature <= 0.5 to avoid "probability tensor contains inf/nan" CUDA assert
                t = float(temperature)
                use_sample = t > 0.5
                gen_kw = dict(
                    **tokenized,
                    max_new_tokens=max_tokens,
                    do_sample=use_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                if use_sample:
                    gen_kw["temperature"] = max(0.01, t)
                with torch.no_grad():
                    outputs = model.generate(**gen_kw)
                
                # Decode
                for j, output_ids in enumerate(outputs):
                    input_length = tokenized['input_ids'][j].shape[0]
                    generated_ids = output_ids[input_length:]
                    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    results.append(response)
            
            print(f"✓ Generated {len(results)} responses from local model")
            
            # Convert to list format (None for failures)
            formatted_results = []
            for result in results:
                if result and result.strip() and result != "{}":
                    formatted_results.append(result.strip())
                else:
                    formatted_results.append(None)
            
            return formatted_results
        except ImportError as e:
            print(f"❌ Local model not available (transformers not installed?): {e}")
            print("   Falling back to API. Install with: pip install transformers torch")
            import traceback
            print(traceback.format_exc())
            # Fall through to API
        except Exception as e:
            print(f"❌ Local model failed: {e}")
            print(f"   Model: {LOCAL_MODEL_NAME}")
            print("   Falling back to API")
            import traceback
            print(traceback.format_exc())
            # Fall through to API
    
    # Use Azure API (original implementation)
    import concurrent.futures
    from typing import List, Optional
    
    def call_single(prompt: str) -> Optional[str]:
        try:
            return call_model(system_prompt, prompt, max_tokens, temperature, enforce_json)
        except ContentFilteredError:
            return None
        except ValueError as e:
            # Re-raise ValueError (like 401 errors with helpful messages) so user sees them
            error_msg = str(e)
            if "401" in error_msg or "Authentication Failed" in error_msg or "USE_LOCAL_MODEL" in error_msg:
                # This is a configuration error that user needs to fix
                print(f"\n❌ Configuration Error: {error_msg}\n")
                raise
            logging.warning(f"Batch call failed: {e}")
            return None
        except Exception as e:
            logging.warning(f"Batch call failed: {e}")
            return None
    
    results = []
    # Process in batches to avoid overwhelming the API
    for i in range(0, len(user_prompts), max_parallel):
        batch = user_prompts[i:i + max_parallel]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            batch_results = list(executor.map(call_single, batch))
        results.extend(batch_results)
    
    return results


# For backward compatibility: export client as a lazy property
# This allows existing code that imports `client` to work
class _ClientProxy:
    """Proxy object that lazily initializes the Azure client when accessed."""
    def __getattr__(self, name):
        client = _get_azure_client()
        if client is None:
            raise ValueError("Azure client not available. Local model may be enabled or credentials missing.")
        return getattr(client, name)

# Export client as a proxy object for backward compatibility
client = _ClientProxy()
