import sys
from pathlib import Path

import torch


PLUGIN_DIR = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_DIR.parents[1]
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(COMFY_ROOT))

from scopes import KKDaVinciScopes, SCOPE_TYPES, render_scope_image  # noqa: E402


def _gradient_image():
    height, width = 96, 144
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height),
        torch.linspace(0.0, 1.0, width),
        indexing="ij",
    )
    image = torch.stack((xx, yy, (1.0 - xx) * (0.25 + yy * 0.75)), dim=-1)
    return image.unsqueeze(0)


def test_all_scope_types_render():
    image = _gradient_image()
    for scope_type in SCOPE_TYPES:
        output = render_scope_image(
            image,
            scope_type,
            width=320,
            height=256,
            intensity=1.25,
            show_grid=True,
            analysis_resolution=128,
        )
        assert output.shape == (1, 256, 320, 3)
        assert bool(torch.isfinite(output).all())
        assert float(output.max()) > 0.25
        assert float(output.std()) > 0.01


def test_batch_is_preserved():
    image = _gradient_image()
    images = torch.cat((image, 1.0 - image), dim=0)
    output = render_scope_image(
        images,
        SCOPE_TYPES[1],
        width=320,
        height=256,
        analysis_resolution=128,
    )
    assert output.shape[0] == 2
    assert float((output[0] - output[1]).abs().mean()) > 0.005


def test_monitor_returns_custom_embedded_ui_payload():
    node = KKDaVinciScopes()
    result = node.render(
        _gradient_image(),
        SCOPE_TYPES[1],
        320,
        256,
        1.25,
        True,
        128,
    )
    assert "result" not in result
    assert "images" not in result["ui"]
    descriptors = result["ui"]["kk_scope"]
    assert len(descriptors) == 1
    assert descriptors[0]["scope_type"] == SCOPE_TYPES[1]
    assert descriptors[0]["scope_width"] == 320


if __name__ == "__main__":
    test_all_scope_types_render()
    test_batch_is_preserved()
    test_monitor_returns_custom_embedded_ui_payload()
    print("All KK DaVinci scope tests passed.")
