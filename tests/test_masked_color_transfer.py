import sys
import types
from pathlib import Path

import torch


PLUGIN_DIR = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_DIR.parents[1]
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(COMFY_ROOT))

# The algorithm tests do not need a running Comfy server. Provide the tiny
# model-management surface used by the node so the test also runs standalone.
try:
    import comfy.model_management  # noqa: F401
except Exception:
    comfy_module = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")
    utils = types.ModuleType("comfy.utils")
    model_management.get_torch_device = lambda: torch.device("cpu")
    model_management.intermediate_device = lambda: torch.device("cpu")
    model_management.intermediate_dtype = lambda: torch.float32
    comfy_module.model_management = model_management
    comfy_module.utils = utils
    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.utils"] = utils

from masked_color_transfer import KKColorRestore  # noqa: E402


def _images():
    torch.manual_seed(9)
    target = torch.rand((2, 40, 48, 3)) * 0.65 + 0.1
    reference = (target * torch.tensor([1.18, 0.82, 0.92]) + torch.tensor([0.03, 0.01, 0.04])).clamp(0, 1)
    return target, reference


def _run(mask=None, invert=False, method="reinhard_lab", reference_mask=None):
    target, reference = _images()
    node = KKColorRestore()
    result, effective_mask = node.transfer(
        target,
        reference,
        method,
        "per_frame",
        0.8,
        invert,
        0,
        0,
        mask,
        reference_mask,
    )
    return target, result.float(), effective_mask.float()


def test_optional_mask_means_full_image():
    target, result, effective_mask = _run()
    assert effective_mask.shape == target.shape[:3]
    assert torch.all(effective_mask == 1)
    assert float((result - target).abs().mean()) > 0.005


def test_mask_changes_only_selected_region():
    mask = torch.zeros((1, 40, 48))
    mask[:, 10:30, 12:36] = 1
    target, result, effective_mask = _run(mask)
    selected = effective_mask.bool().unsqueeze(-1).expand_as(target)
    assert float((result[~selected] - target[~selected]).abs().max()) == 0.0
    assert float((result[selected] - target[selected]).abs().mean()) > 0.005


def test_invert_mask_swaps_selected_region():
    mask = torch.zeros((1, 40, 48))
    mask[:, 10:30, 12:36] = 1
    target, result, effective_mask = _run(mask, invert=True)
    protected = mask.expand(2, -1, -1).bool().unsqueeze(-1).expand_as(target)
    assert float((result[protected] - target[protected]).abs().max()) == 0.0
    assert torch.all(effective_mask[:, 10:30, 12:36] == 0)
    assert float((result[~protected] - target[~protected]).abs().mean()) > 0.005


def test_invert_without_mask_still_processes_full_image():
    target, result, effective_mask = _run(mask=None, invert=True)
    assert torch.all(effective_mask == 1)
    assert float((result - target).abs().mean()) > 0.005


def test_reference_mask_drives_local_statistics():
    target = torch.full((1, 40, 48, 3), 0.15)
    reference = torch.full((1, 40, 48, 3), 0.85)
    target[:, 5:20, 6:22] = torch.tensor([0.55, 0.25, 0.18])
    reference[:, 20:35, 25:42] = torch.tensor([0.20, 0.58, 0.34])
    target_mask = torch.zeros((1, 40, 48))
    target_mask[:, 5:20, 6:22] = 1
    reference_mask = torch.zeros((1, 40, 48))
    reference_mask[:, 20:35, 25:42] = 1

    node = KKColorRestore()
    result, _ = node.transfer(
        target,
        reference,
        "reinhard_lab",
        "per_frame",
        1.0,
        False,
        0,
        0,
        target_mask,
        reference_mask,
    )
    selected_mean = result[0, 5:20, 6:22].float().mean(dim=(0, 1))
    expected = torch.tensor([0.20, 0.58, 0.34])
    assert float((selected_mean - expected).abs().max()) < 0.002


def test_all_methods_execute():
    mask = torch.ones((1, 40, 48))
    for method in ("reinhard_lab", "mkl_lab", "histogram"):
        target, result, _ = _run(mask, method=method)
        assert result.shape == target.shape
        assert bool(torch.isfinite(result).all())


if __name__ == "__main__":
    test_optional_mask_means_full_image()
    test_mask_changes_only_selected_region()
    test_invert_mask_swaps_selected_region()
    test_invert_without_mask_still_processes_full_image()
    test_reference_mask_drives_local_statistics()
    test_all_methods_execute()
    print("All masked color transfer tests passed.")
