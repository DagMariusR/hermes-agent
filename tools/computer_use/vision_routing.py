"""Capture routing for ``computer_use`` screenshots: multimodal envelope, or pre-analyse via aux vision?

The shared gate ``tools.vision_tools._native_tool_result_images`` (image routing resolves ``native`` AND the route
accepts tool-result images) decides, with one extra hard veto on top: the provider TRANSPORT must carry images inside
tool-result messages. The shared gate ORs transport with the capability catalog, so a vision-capable model behind a
provider that cannot carry tool-result media passes it — precisely the hard 400/404 this lane exists to stop. The
veto outranks a user-declared ``supports_vision: true`` too.

``auxiliary.vision`` never wins the decision: it is the fallback describer for the aux path. Routing every capture
through aux requires ``agent.image_input_mode: text``.
"""

from __future__ import annotations

from typing import Any, Optional


def should_route_capture_to_aux_vision(provider: str, model: str, cfg: Optional[dict[str, Any]]) -> bool:
    """True iff the screenshot should be pre-analysed via aux vision; False keeps the multimodal envelope. Fails
    closed toward aux: a screenshot sent to a model that cannot read it is a hard failure, aux costs one LLM call."""
    from tools.vision_tools import _native_tool_result_images, _supports_media_in_tool_results
    if not _supports_media_in_tool_results(provider, model):  # transport veto (profile veto included)
        return True
    return not _native_tool_result_images(provider, model, cfg)


__all__ = ["should_route_capture_to_aux_vision"]
