"""Vision-routing decisions for ``computer_use`` capture results. ``capture`` (mode som|vision) returns a
``_multimodal`` screenshot envelope as the tool result. A text-only main model, or a provider that rejects multimodal
tool results, turns that into a hard 400/404 — even with a working ``auxiliary.vision`` model in config. This module
decides: multimodal envelope, or pre-analyse via aux vision so the main model only ever sees text?

Decision order (the provider tool-result gate outranks a user-declared True, mirroring the ``vision_analyze``
profile veto):
1. User-declared ``supports_vision=False`` for the active route (escape hatch for custom/local VLMs) → aux routing
   immediately.
2. Provider+model does not carry images inside tool-result messages → aux routing, even against a user-declared True.
3. User-declared ``supports_vision=True`` → multimodal (tool-result support already proven).
4. Catalog ``supports_vision`` resolves True → multimodal; everything else (non-vision model, lookup failure) → aux routing.

``auxiliary.vision`` never wins the decision: it is the fallback describer for the aux path. Routing every capture
through aux requires ``agent.image_input_mode: text``.

Fails *closed* toward aux routing when metadata is missing or ambiguous: a screenshot sent to a model that cannot read
it is a hard failure, while aux routing costs one extra LLM call and yields a usable description.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def _lookup_user_declared_supports_vision(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Config-declared ``supports_vision`` for the active route (None on failure)."""
    try:
        from agent.image_routing import _supports_vision_override
        return _supports_vision_override(cfg, provider, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use vision_routing: config override lookup failed: %s", exc)
        return None

def _models_dev_supports_vision(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Raw models.dev capability lookup — fallback when ``agent.image_routing`` is unavailable."""
    from agent.models_dev import get_model_capabilities
    caps = get_model_capabilities(provider, model)
    return None if caps is None else caps.supports_vision

def _lookup_supports_vision(provider: str, model: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[bool]:
    """Config/models.dev ``supports_vision`` for *(provider, model)*; prefers
    ``agent.image_routing._lookup_supports_vision``. Any lookup error → None (caller fails closed to aux)."""
    if not provider or not model:
        return None
    try:
        from agent.image_routing import _lookup_supports_vision as lookup
    except Exception:
        lookup = _models_dev_supports_vision
    try:
        return lookup(provider, model, cfg)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use vision_routing: caps lookup failed for %s:%s — %s", provider, model, exc)
        return None

def _provider_accepts_multimodal_tool_result(provider: str, model: str) -> Optional[bool]:
    """TRANSPORT only: may *provider*+*model* carry images inside tool-result messages? Deliberately NOT
    ``tools.vision_tools._accepts_tool_result_images``, which ORs this with the capability catalog — a vision-capable
    model behind a provider that cannot carry tool-result media is precisely the hard 400/404 this gate exists to
    stop, and the OR lets it through. The profile veto is honoured. None on import failure so callers fall back to
    aux, not guess."""
    if not provider:
        return None
    try:
        from tools.vision_tools import _profile_rejects_tool_media, _supports_media_in_tool_results
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use vision_routing: tool-result support lookup failed: %s", exc)
        return None
    if _profile_rejects_tool_media(provider, model):
        return False
    return bool(_supports_media_in_tool_results(provider, model))

def should_route_capture_to_aux_vision(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> bool:
    """True iff the screenshot should be pre-analysed via aux vision; False keeps the multimodal envelope. *provider* is
    the lower-case canonical id, *model* the slug sent to the provider, *cfg* the loaded ``config.yaml`` dict (or None).
    Steps follow the module docstring's decision order: a user-declared ``supports_vision=False`` routes aux immediately;
    then a provider that rejects multimodal tool results routes aux before any user-declared ``True`` can win; then a
    user-declared ``supports_vision=True`` keeps the multimodal envelope; otherwise the caps catalog must resolve
    ``supports_vision=True``. An explicit ``auxiliary.vision`` backend is the fallback describer for the aux path, never
    a routing trigger."""
    user_declared = _lookup_user_declared_supports_vision(provider, model, cfg)
    if user_declared is False:
        return True
    if not _provider_accepts_multimodal_tool_result(provider, model):
        return True
    if user_declared is True:
        return False
    return _lookup_supports_vision(provider, model, cfg) is not True

__all__ = ["should_route_capture_to_aux_vision"]
