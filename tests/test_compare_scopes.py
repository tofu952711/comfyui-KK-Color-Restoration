import sys
from pathlib import Path

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))

from custom_nodes.comfyui_masked_color_transfer.compare_scopes import (  # noqa: E402
    COMPARE_VIEWS,
    KKDualImageColorScopes,
    apply_primary_adjustments,
)
from custom_nodes.comfyui_masked_color_transfer.auto_grade import (  # noqa: E402
    PARAMETER_NAMES,
    auto_grade_to_reference,
)
from custom_nodes.comfyui_masked_color_transfer.scopes import SCOPE_TYPES  # noqa: E402


def _images():
    torch.manual_seed(21)
    target = torch.rand((1, 72, 96, 3)) * 0.65 + 0.12
    reference = (target * torch.tensor([1.12, 0.86, 0.94]) + torch.tensor([0.025, 0.01, 0.04])).clamp(0, 1)
    return reference, target


def test_automatic_parameter_solver_reduces_distribution_error():
    reference, target = _images()
    grade = auto_grade_to_reference(reference, target, 1.0, 96, steps=28)
    assert grade.after_error < grade.before_error * 0.75
    assert grade.improvement > 25.0
    assert set(grade.parameters) == set(PARAMETER_NAMES)
    assert float((grade.image - target).abs().mean()) > 1e-4


def test_primary_numeric_adjustments_change_image():
    _, target = _images()
    brighter = apply_primary_adjustments(target, exposure_ev=1.0)
    assert float(brighter.mean()) > float(target.mean())
    warmer = apply_primary_adjustments(target, temperature=60.0)
    assert float(warmer[..., 0].mean() - warmer[..., 2].mean()) > float(
        target[..., 0].mean() - target[..., 2].mean()
    )


def test_node_returns_adjusted_image_and_embedded_compare_payload():
    reference, target = _images()
    node = KKDualImageColorScopes()
    result = node.compare_and_adjust(
        reference,
        target,
        SCOPE_TYPES[1],
        COMPARE_VIEWS[0],
        True,
        0.8,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        128,
    )
    adjusted = result["result"][0].float()
    pairs = result["ui"]["kk_compare_scope"]
    assert adjusted.shape == target.shape
    assert float((adjusted - target).abs().mean()) > 1e-4
    assert len(pairs) == 1
    assert pairs[0]["scope_type"] == SCOPE_TYPES[1]
    assert "difference" in pairs[0]
    assert set(pairs[0]["metrics"]) == {"reference", "target", "delta", "suggestions", "adjustment"}
    assert pairs[0]["metrics"]["adjustment"]["auto_match"] is True
    assert pairs[0]["metrics"]["adjustment"]["mean_pixel_change"] > 0.0
    assert pairs[0]["metrics"]["adjustment"]["match_score"] > 0.0
    assert set(pairs[0]["metrics"]["adjustment"]["parameters"]) == set(PARAMETER_NAMES)
    assert "images" not in result["ui"]
    assert len(result["result"]) == 3
    assert "KK Auto Grade v1" in result["result"][1]
    assert result["result"][2] > 0.0
    assert KKDualImageColorScopes.RETURN_TYPES == ("IMAGE", "STRING", "FLOAT")
    assert KKDualImageColorScopes.RETURN_NAMES == ("image2_adjusted", "grade_parameters", "match_score")


if __name__ == "__main__":
    test_automatic_parameter_solver_reduces_distribution_error()
    test_primary_numeric_adjustments_change_image()
    test_node_returns_adjusted_image_and_embedded_compare_payload()
    print("All KK dual-image color scope tests passed.")
