"""Bundled control signals: mention behavior is fixed per response_mode."""

from typing import Any, Mapping, Optional


def mention_level_for_response_mode(response_mode: Optional[Any]) -> str:
    """
    Canonical mention level for each response_mode (single policy field).

    - direct_answer: answer as if the safe prompt were the question; no meta about reframing.
    - soft_reframe: light coherence bridge; still no accusation or "disallowed" framing.
    - acknowledge_and_redirect: clear, neutral limitation then helpful redirect.
    """
    if response_mode is None:
        return "none"
    m = str(response_mode).strip().lower()
    if m == "direct_answer":
        return "none"
    if m == "soft_reframe":
        return "brief"
    if m == "acknowledge_and_redirect":
        return "explicit"
    return "none"


def apply_mention_level_to_control_signals(control: Mapping[str, Any]) -> dict:
    """Return a copy of control with mention_level set from response_mode."""
    out = dict(control)
    mode = out.get("response_mode")
    if mode is not None and str(mode).strip():
        out["mention_level"] = mention_level_for_response_mode(mode)
    return out


def effective_mention_level(rec: Mapping[str, Any]) -> str:
    """Mention level for generation: always derived from response_mode."""
    return mention_level_for_response_mode(rec.get("response_mode"))
