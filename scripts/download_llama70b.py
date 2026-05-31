#!/usr/bin/env python3
"""
Download Llama 70B model from Hugging Face.

This script downloads the Llama 70B model to your local Hugging Face cache.
The model will be automatically used when you set LOCAL_MODEL or LOCAL_JUDGE_MODEL
in your .env file.

Available Llama 70B models:
- meta-llama/Llama-2-70b-chat-hf (recommended for chat/instruct tasks)
- meta-llama/Llama-2-70b-hf (base model)
- meta-llama/Llama-3-70b-Instruct (if available, newer version)
"""

import os
import sys
from pathlib import Path

def check_huggingface_login():
    """Check if user is logged into Hugging Face."""
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✓ Logged into Hugging Face as: {user.get('name', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ Not logged into Hugging Face: {e}")
        print("\nTo download Llama models, you need to:")
        print("1. Request access at: https://huggingface.co/meta-llama/Llama-2-70b-chat-hf")
        print("2. Login with: huggingface-cli login")
        print("   Or set HF_TOKEN environment variable")
        return False

def download_model(model_name: str, cache_dir: str = None, use_symlinks: bool = False):
    """
    Download a model from Hugging Face.
    
    Args:
        model_name: Hugging Face model identifier
        cache_dir: Custom cache directory (default: uses HF_HOME or ~/.cache/huggingface)
        use_symlinks: Use symlinks instead of copying files (saves disk space)
    """
    try:
        from huggingface_hub import snapshot_download
        import torch
    except ImportError:
        print("❌ Missing required packages. Install with:")
        print("   pip install huggingface_hub transformers torch")
        sys.exit(1)
    
    # Determine cache directory
    if cache_dir is None:
        # Try to get from huggingface_hub constants (uses HF_HOME if set)
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_dir = HF_HUB_CACHE
        except (ImportError, AttributeError):
            # Fallback: Check HF_HOME environment variable
            hf_home = os.getenv("HF_HOME")
            if hf_home:
                cache_dir = os.path.join(hf_home, "hub")
            else:
                # Default to ~/.cache/huggingface/hub
                cache_dir = os.path.join(Path.home(), ".cache", "huggingface", "hub")
    
    # Expand user path and create directory if needed
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"\n📥 Downloading {model_name}...")
    print(f"   This may take a while (model is ~140GB)...")
    print(f"   Cache directory: {cache_dir}\n")
    
    try:
        # Download model files (will use default cache location automatically)
        # The cache_dir parameter ensures it goes to the right place
        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_files_only=False,
            resume_download=True,
            local_dir_use_symlinks=use_symlinks,
            cache_dir=cache_dir
        )
        
        print(f"\n✓ Model downloaded successfully!")
        print(f"   Cache location: {downloaded_path}")
        print(f"\n💡 To use this model, add to your .env file:")
        print(f"   LOCAL_MODEL={model_name}")
        print(f"   LOCAL_JUDGE_MODEL={model_name}")
        print(f"   USE_LOCAL_MODEL=true")
        print(f"   USE_LOCAL_JUDGE=true")
        print(f"\n💡 To use this cache location permanently, set in your .env or .bashrc:")
        print(f"   export HF_HOME={os.path.dirname(cache_dir)}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower():
            print(f"\n❌ Authentication failed. You need to:")
            print(f"   1. Request access at: https://huggingface.co/{model_name}")
            print(f"   2. Login with: huggingface-cli login")
            print(f"   3. Or set HF_TOKEN environment variable")
        elif "403" in error_msg or "forbidden" in error_msg.lower():
            print(f"\n❌ Access denied. You need to:")
            print(f"   1. Request access at: https://huggingface.co/{model_name}")
            print(f"   2. Wait for Meta's approval")
            print(f"   3. Then try again")
        else:
            print(f"\n❌ Download failed: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Llama 70B model from Hugging Face"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-70b-chat-hf",
        help="Model to download (default: meta-llama/Llama-2-70b-chat-hf)"
    )
    parser.add_argument(
        "--use-symlinks",
        action="store_true",
        help="Use symlinks instead of copying files (saves disk space)"
    )
    parser.add_argument(
        "--skip-login-check",
        action="store_true",
        help="Skip Hugging Face login check"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory (default: uses HF_HOME or ~/.cache/huggingface/hub)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Llama 70B Model Downloader")
    print("=" * 70)
    
    # Check login (unless skipped)
    if not args.skip_login_check:
        if not check_huggingface_login():
            print("\n⚠️  Continuing anyway (use --skip-login-check to suppress this)")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Download model
    success = download_model(args.model, args.cache_dir, args.use_symlinks)
    
    if success:
        print("\n" + "=" * 70)
        print("✓ Download complete!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ Download failed. See errors above.")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()

