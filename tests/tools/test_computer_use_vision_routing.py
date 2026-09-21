"""Unit tests for tools.computer_use.vision_routing.

Cover the small ``should_route_capture_to_aux_vision`` policy helper that
decides whether a captured screenshot from ``computer_use(action='capture')``
should be returned as a multimodal envelope (main model handles vision
natively) or pre-analysed via the ``auxiliary.vision`` pipeline so the
main model only sees text.

Contract: ``auxiliary.vision`` is the fallback describer for the aux path —
it never wins the decision. Decision order: a user-declared
``supports_vision=False`` routes aux immediately, then required tool-result
support gates ahead of any user-declared ``True``, then the user-declared
``True`` keeps the envelope, then a required ``supports_vision=True``;
routing every capture through aux requires ``agent.image_input_mode: text``.

The companion end-to-end regression for #24015 lives in
``tests/tools/test_computer_use_capture_routing.py``; this file pins the
unit contract of the helper in isolation so behaviour does not regress
silently if the surrounding ``computer_use`` plumbing is refactored.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# should_route_capture_to_aux_vision
# ---------------------------------------------------------------------------

class TestRouteDecision:
    """End-to-end policy: user-declared False > tool-result support > user-declared True > vision caps."""

    def test_explicit_aux_config_does_not_win(self):
        """Reserve contract: an explicit ``auxiliary.vision`` backend is the
        fallback describer, never a routing trigger. A vision-capable main
        model with working tool-result support keeps the multimodal envelope;
        routing every capture through aux requires ``agent.image_input_mode:
        text``."""
        from tools.computer_use import vision_routing

        cfg = {
            "auxiliary": {
                "vision": {
                    "provider": "openrouter",
                    "model": "google/gemini-2.5-flash",
                }
            }
        }
        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=True):
            assert vision_routing.should_route_capture_to_aux_vision(
                "anthropic", "claude-opus-4-5", cfg
            ) is False

    def test_non_vision_main_model_routes_to_aux(self):
        """The reported #24015 scenario: tencent/hy3-preview has no vision."""
        from tools.computer_use import vision_routing

        cfg = {"model": {"default": "tencent/hy3-preview", "provider": "openrouter"}}
        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=False):
            assert vision_routing.should_route_capture_to_aux_vision(
                "openrouter", "tencent/hy3-preview", cfg
            ) is True

    def test_user_declared_false_routes_to_aux(self):
        """``supports_vision: False`` routes aux even when the provider would
        accept multimodal tool results."""
        from tools.computer_use import vision_routing

        cfg = {"model": {"supports_vision": False}}
        with patch.object(vision_routing, "_lookup_supports_vision", return_value=True), \
             patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=True):
            assert vision_routing.should_route_capture_to_aux_vision(
                "custom", "text-only-model", cfg
            ) is True

    def test_vision_main_model_no_override_keeps_multimodal(self):
        """Default path: vision-capable main model + no aux override → native."""
        from tools.computer_use import vision_routing

        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=True):
            assert vision_routing.should_route_capture_to_aux_vision(
                "anthropic", "claude-opus-4-5", None
            ) is False


    def test_provider_rejecting_tool_results_routes_aux_even_when_user_declared_true(self):
        """The provider tool-result gate outranks a user-declared ``True``:
        without multimodal tool-result support the envelope would hard-fail
        downstream, so it routes aux even for a config-declared vision model."""
        from tools.computer_use import vision_routing

        cfg = {
            "model": {
                "default": "Qwen3.6-35B-A3B-local-vlm",
                "provider": "omlx",
                "supports_vision": True,
            }
        }
        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=False):
            assert vision_routing.should_route_capture_to_aux_vision(
                "custom", "Qwen3.6-35B-A3B-local-vlm", cfg
            ) is True

    def test_user_declared_true_with_provider_support_keeps_native(self):
        """With tool-result support proven, a user-declared ``supports_vision:
        True`` keeps the multimodal envelope without consulting the catalog
        (the ``_lookup_supports_vision`` False stub fails the test if the
        decision wrongly falls through to the catalog lookup)."""
        from tools.computer_use import vision_routing

        cfg = {"model": {"supports_vision": True}}
        with patch.object(vision_routing, "_lookup_supports_vision", return_value=False), \
             patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=True):
            assert vision_routing.should_route_capture_to_aux_vision(
                "custom", "local-vlm", cfg
            ) is False


    def test_unknown_provider_capabilities_fail_closed(self):
        """When tool-result lookup returns None, route to aux (safe default)."""
        from tools.computer_use import vision_routing

        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=None):
            assert vision_routing.should_route_capture_to_aux_vision(
                "exotic-provider", "exotic-model", {}
            ) is True


    def test_unknown_caps_fail_closed_even_with_explicit_aux_config(self):
        """An explicit aux.vision block neither wins nor rescues: with every
        capability lookup unknown the fail-closed default (aux) still applies,
        because the screenshot cannot be proven safe to attach."""
        from tools.computer_use import vision_routing

        cfg = {"auxiliary": {"vision": {"provider": "openrouter"}}}
        with patch.object(vision_routing,
                          "_provider_accepts_multimodal_tool_result",
                          return_value=None):
            assert vision_routing.should_route_capture_to_aux_vision(
                "openrouter", "tencent/hy3-preview", cfg
            ) is True


# ---------------------------------------------------------------------------
# Internal lookups — defensive paths
# ---------------------------------------------------------------------------

class TestLookupHelpers:


    def test_provider_accepts_multimodal_tool_result_returns_none_for_blank_provider(self):
        from tools.computer_use.vision_routing import (
            _provider_accepts_multimodal_tool_result,
        )
        assert _provider_accepts_multimodal_tool_result("", "claude") is None


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------

class TestModuleSurface:
    """Pin the public surface so dependents stay in lockstep."""

    def test_should_route_capture_to_aux_vision_is_exported(self):
        from tools.computer_use import vision_routing

        assert "should_route_capture_to_aux_vision" in vision_routing.__all__
        assert callable(vision_routing.should_route_capture_to_aux_vision)

    @pytest.mark.parametrize("name", [
        "_lookup_supports_vision",
        "_provider_accepts_multimodal_tool_result",
    ])
    def test_internal_helpers_are_addressable(self, name):
        """Internal helpers stay importable so tests can monkeypatch them."""
        from tools.computer_use import vision_routing

        assert hasattr(vision_routing, name)
        assert callable(getattr(vision_routing, name))


class TestGateAgreementWithVisionAnalyze:
    """#115248 asked for one predicate behind both lanes so the route never depends on which tool asked.
    The capture lane keeps that goal but resolves the disagreements in the *fail-closed* direction, which
    ``tools.vision_tools._accepts_tool_result_images`` does not: that helper ORs the transport whitelist with
    the capability catalog, so it answers True when only one of the two is known. Sending a screenshot the
    receiver cannot read is a hard 400/404; one extra aux call is not. The module docstring already promises
    to fail closed on missing or ambiguous metadata — these tests pin that promise for the capture lane."""

    def test_catalog_vision_model_off_the_provider_whitelist_routes_to_aux(self):
        """Catalog says the model sees, but the provider cannot carry images inside tool results. Transport is
        an inviolable gate: the envelope would 400 regardless of what the model can do."""
        from tools.computer_use import vision_routing

        cfg = {"agent": {"image_input_mode": "native"}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=True), \
             patch("tools.vision_tools._supports_media_in_tool_results", return_value=False), \
             patch("tools.vision_tools._profile_rejects_tool_media", return_value=False):
            assert vision_routing.should_route_capture_to_aux_vision("deepseek", "deepseek-flash", cfg) is True

    def test_profile_veto_still_routes_a_catalog_vision_model_to_aux(self):
        from tools.computer_use import vision_routing

        with patch("agent.image_routing._lookup_supports_vision", return_value=True), \
             patch("tools.vision_tools._supports_media_in_tool_results", return_value=False), \
             patch("tools.vision_tools._profile_rejects_tool_media", return_value=True):
            assert vision_routing.should_route_capture_to_aux_vision("xiaomi", "mimo-v2.5", {}) is True

    def test_whitelisted_provider_with_catalog_unknown_model_routes_to_aux(self):
        """Transport is proven, capability is unknown (a proxy alias absent from models.dev and config).
        ``_accepts_tool_result_images`` answers True here on transport alone, so ``vision_analyze`` embeds
        natively while capture routes to aux — the two lanes DO diverge in this case, deliberately.
        "Unknown" is not "yes": an unrecognised alias may well be a text-only model, and the capture lane
        will not gamble a hard failure on it. The durable fix is to make the shared gate AND its two inputs
        instead of ORing them, which would pull ``vision_analyze`` into the same fail-closed posture."""
        from tools.computer_use import vision_routing
        from tools.vision_tools import _accepts_tool_result_images

        cfg = {"agent": {"image_input_mode": "native"}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=None), \
             patch("tools.vision_tools._supports_media_in_tool_results", return_value=True), \
             patch("tools.vision_tools._profile_rejects_tool_media", return_value=False):
            assert _accepts_tool_result_images("anthropic", "my-proxy-claude", cfg) is True
            assert vision_routing.should_route_capture_to_aux_vision("anthropic", "my-proxy-claude", cfg) is True
