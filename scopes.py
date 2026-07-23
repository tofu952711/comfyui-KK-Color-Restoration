"""DaVinci Resolve inspired video scopes for ComfyUI."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nodes import PreviewImage


SCOPE_TYPES = (
    "分量图 (RGB Parade)",
    "波形图 (RGB Waveform)",
    "矢量图 (Vectorscope)",
    "直方图 (Histogram)",
    "CIE色度图 (CIE 1931)",
)


def _analysis_image(image: torch.Tensor, max_side: int) -> torch.Tensor:
    """Downsample one BHWC frame for fast scope analysis."""
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale >= 1.0:
        return image.to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    out_h = max(1, round(height * scale))
    out_w = max(1, round(width * scale))
    resized = F.interpolate(
        image.permute(2, 0, 1).unsqueeze(0).to(device="cpu", dtype=torch.float32),
        size=(out_h, out_w),
        mode="area",
    )
    return resized.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)


def _base_canvas(height: int, width: int) -> torch.Tensor:
    color = torch.tensor([0.006, 0.008, 0.012], dtype=torch.float32)
    return color.view(1, 1, 3).expand(height, width, 3).clone()


def _screen(canvas: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 - canvas) * (1.0 - trace.clamp(0.0, 1.0))


def _draw_line(canvas: torch.Tensor, p0, p1, color, alpha: float = 1.0, thickness: int = 1):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    steps = max(2, int(max(abs(x1 - x0), abs(y1 - y0))) + 1)
    xs = torch.linspace(x0, x1, steps).round().long().clamp(0, canvas.shape[1] - 1)
    ys = torch.linspace(y0, y1, steps).round().long().clamp(0, canvas.shape[0] - 1)
    line_color = torch.tensor(color, dtype=canvas.dtype).view(1, 3)
    for offset in range(-(thickness // 2), thickness - thickness // 2):
        yy = (ys + offset).clamp(0, canvas.shape[0] - 1)
        old = canvas[yy, xs]
        canvas[yy, xs] = old * (1.0 - alpha) + line_color * alpha


def _draw_grid(canvas: torch.Tensor, vertical_divisions: int = 10, horizontal_divisions: int = 8):
    height, width = canvas.shape[:2]
    minor = (0.18, 0.145, 0.025)
    major = (0.42, 0.33, 0.035)
    for i in range(vertical_divisions + 1):
        x = round(i * (width - 1) / vertical_divisions)
        _draw_line(canvas, (x, 0), (x, height - 1), major if i in (0, vertical_divisions) else minor, 0.65)
    for i in range(horizontal_divisions + 1):
        y = round(i * (height - 1) / horizontal_divisions)
        is_major = i in (0, horizontal_divisions) or i == horizontal_divisions // 2
        _draw_line(canvas, (0, y), (width - 1, y), major if is_major else minor, 0.75)


def _normalize_density(values: torch.Tensor, intensity: float) -> torch.Tensor:
    logged = torch.log1p(values)
    nonzero = logged[logged > 0]
    if nonzero.numel() == 0:
        return logged
    reference = torch.quantile(nonzero, 0.995).clamp_min(1e-6)
    return (logged / reference * float(intensity)).clamp(0.0, 1.0)


def _waveform(frame: torch.Tensor, height: int, width: int, intensity: float, grid: bool, parade: bool):
    canvas = _base_canvas(height, width)
    if grid:
        _draw_grid(canvas, vertical_divisions=12 if parade else 10, horizontal_divisions=8)

    source_h, source_w = frame.shape[:2]
    x_source = torch.arange(source_w).view(1, source_w).expand(source_h, source_w)
    density = torch.zeros((3, height * width), dtype=torch.float32)
    margin = max(4, width // 100)

    for channel in range(3):
        if parade:
            section = (width - margin * 4) / 3.0
            x = margin + channel * (section + margin) + x_source.float() / max(source_w - 1, 1) * (section - 1)
        else:
            x = x_source.float() / max(source_w - 1, 1) * (width - 1)
        y = (1.0 - frame[..., channel]) * (height - 1)
        index = y.round().long().clamp(0, height - 1) * width + x.round().long().clamp(0, width - 1)
        density[channel].scatter_add_(0, index.reshape(-1), torch.ones(index.numel()))

    trace = _normalize_density(density, intensity).reshape(3, height, width).permute(1, 2, 0)
    canvas = _screen(canvas, trace)

    if parade:
        section = (width - margin * 4) / 3.0
        for channel, color in enumerate(((0.9, 0.08, 0.05), (0.05, 0.9, 0.12), (0.08, 0.2, 1.0))):
            x0 = round(margin + channel * (section + margin))
            x1 = round(x0 + section - 1)
            _draw_line(canvas, (x0, 0), (x0, height - 1), color, 0.35)
            _draw_line(canvas, (x1, 0), (x1, height - 1), color, 0.35)
    return canvas


def _histogram(frame: torch.Tensor, height: int, width: int, intensity: float, grid: bool):
    canvas = _base_canvas(height, width)
    if grid:
        _draw_grid(canvas, vertical_divisions=8, horizontal_divisions=8)

    channels = []
    for channel in range(3):
        hist = torch.histc(frame[..., channel], bins=256, min=0.0, max=1.0)
        hist = torch.log1p(hist)
        hist = hist / hist.max().clamp_min(1e-6)
        hist = F.interpolate(hist.view(1, 1, -1), size=width, mode="linear", align_corners=True).view(-1)
        channels.append(hist)
    curves = torch.stack(channels).clamp(0.0, 1.0)

    rows = torch.arange(height).view(height, 1)
    trace = torch.zeros((height, width, 3), dtype=torch.float32)
    for channel in range(3):
        top = ((1.0 - curves[channel]) * (height - 1)).round().long().view(1, width)
        fill = (rows >= top).float() * min(0.75, 0.22 * float(intensity))
        line_y = top.squeeze(0).clamp(0, height - 1)
        trace[..., channel] = fill
        trace[line_y, torch.arange(width), channel] = min(1.0, 0.75 * float(intensity))
    return _screen(canvas, trace)


def _scatter_scope(canvas: torch.Tensor, x: torch.Tensor, y: torch.Tensor, colors: torch.Tensor, intensity: float):
    height, width = canvas.shape[:2]
    x = x.round().long().clamp(0, width - 1)
    y = y.round().long().clamp(0, height - 1)
    index = y * width + x
    count = torch.zeros(height * width, dtype=torch.float32)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float32))
    color_sum = torch.zeros((3, height * width), dtype=torch.float32)
    for channel in range(3):
        color_sum[channel].scatter_add_(0, index, colors[:, channel])
    average = color_sum / count.clamp_min(1.0).unsqueeze(0)
    density = _normalize_density(count, intensity).unsqueeze(0)
    trace = (average * 0.82 + 0.18) * density
    trace = trace.reshape(3, height, width).permute(1, 2, 0)
    return _screen(canvas, trace)


def _vectorscope(frame: torch.Tensor, height: int, width: int, intensity: float, grid: bool):
    canvas = _base_canvas(height, width)
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    radius = min(height, width) * 0.43
    if grid:
        grid_color = (0.16, 0.23, 0.19)
        _draw_line(canvas, (center_x - radius, center_y), (center_x + radius, center_y), grid_color, 0.8)
        _draw_line(canvas, (center_x, center_y - radius), (center_x, center_y + radius), grid_color, 0.8)
        for ring in (0.25, 0.5, 0.75, 1.0):
            theta = torch.linspace(0, 2 * torch.pi, 361)
            xs = center_x + torch.cos(theta) * radius * ring
            ys = center_y + torch.sin(theta) * radius * ring
            for i in range(360):
                _draw_line(canvas, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), grid_color, 0.52)

    rgb = frame.reshape(-1, 3)
    red, green, blue = rgb.unbind(dim=1)
    cb = -0.114572 * red - 0.385428 * green + 0.5 * blue
    cr = 0.5 * red - 0.454153 * green - 0.045847 * blue
    x = center_x + cb * radius * 2.0
    y = center_y - cr * radius * 2.0
    canvas = _scatter_scope(canvas, x, y, rgb, intensity)

    targets = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
    )
    for color in targets:
        r, g, b = color
        tx = center_x + (-0.114572 * r - 0.385428 * g + 0.5 * b) * radius * 2.0
        ty = center_y - (0.5 * r - 0.454153 * g - 0.045847 * b) * radius * 2.0
        size = max(2, round(radius * 0.018))
        x0, x1 = int(tx - size), int(tx + size)
        y0, y1 = int(ty - size), int(ty + size)
        _draw_line(canvas, (x0, y0), (x1, y0), color, 0.9)
        _draw_line(canvas, (x1, y0), (x1, y1), color, 0.9)
        _draw_line(canvas, (x1, y1), (x0, y1), color, 0.9)
        _draw_line(canvas, (x0, y1), (x0, y0), color, 0.9)
    return canvas


def _cie_to_canvas(x, y, height: int, width: int, margin: int):
    plot_w = width - margin * 2
    plot_h = height - margin * 2
    px = margin + x / 0.8 * plot_w
    py = height - margin - y / 0.9 * plot_h
    return px, py


def _cie_scope(frame: torch.Tensor, height: int, width: int, intensity: float, grid: bool):
    canvas = _base_canvas(height, width)
    margin = max(18, min(height, width) // 18)
    if grid:
        grid_color = (0.12, 0.17, 0.15)
        for value in torch.arange(0.0, 0.81, 0.1):
            x, _ = _cie_to_canvas(value, torch.tensor(0.0), height, width, margin)
            _draw_line(canvas, (x, margin), (x, height - margin), grid_color, 0.75)
        for value in torch.arange(0.0, 0.91, 0.1):
            _, y = _cie_to_canvas(torch.tensor(0.0), value, height, width, margin)
            _draw_line(canvas, (margin, y), (width - margin, y), grid_color, 0.75)

    rgb = frame.reshape(-1, 3)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    red, green, blue = linear.unbind(dim=1)
    xyz_x = 0.4124564 * red + 0.3575761 * green + 0.1804375 * blue
    xyz_y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    xyz_z = 0.0193339 * red + 0.1191920 * green + 0.9503041 * blue
    total = (xyz_x + xyz_y + xyz_z).clamp_min(1e-8)
    chroma_x = xyz_x / total
    chroma_y = xyz_y / total
    px, py = _cie_to_canvas(chroma_x, chroma_y, height, width, margin)
    canvas = _scatter_scope(canvas, px, py, rgb, intensity)

    # Approximate CIE 1931 spectral locus, sufficient for a scope reference.
    locus = (
        (0.1741, 0.0050), (0.1738, 0.0049), (0.1699, 0.0090), (0.1440, 0.0297),
        (0.0913, 0.1327), (0.0454, 0.2950), (0.0082, 0.5384), (0.0139, 0.7502),
        (0.0743, 0.8338), (0.1547, 0.8059), (0.2296, 0.7543), (0.3016, 0.6923),
        (0.3731, 0.6245), (0.4441, 0.5547), (0.5125, 0.4866), (0.5752, 0.4242),
        (0.6270, 0.3725), (0.6658, 0.3340), (0.6915, 0.3083), (0.7080, 0.2920),
        (0.7240, 0.2760), (0.7347, 0.2653), (0.7347, 0.2653), (0.1741, 0.0050),
    )
    locus_points = [_cie_to_canvas(torch.tensor(x), torch.tensor(y), height, width, margin) for x, y in locus]
    for p0, p1 in zip(locus_points[:-1], locus_points[1:]):
        _draw_line(canvas, p0, p1, (0.66, 0.72, 0.69), 0.9)

    primaries = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06), (0.64, 0.33))
    gamut = [_cie_to_canvas(torch.tensor(x), torch.tensor(y), height, width, margin) for x, y in primaries]
    gamut_colors = ((1.0, 0.15, 0.1), (0.15, 1.0, 0.2), (0.15, 0.3, 1.0))
    for i in range(3):
        _draw_line(canvas, gamut[i], gamut[i + 1], gamut_colors[i], 0.85, thickness=2)
    white = _cie_to_canvas(torch.tensor(0.3127), torch.tensor(0.3290), height, width, margin)
    _draw_line(canvas, (white[0] - 4, white[1]), (white[0] + 4, white[1]), (1, 1, 1), 0.9)
    _draw_line(canvas, (white[0], white[1] - 4), (white[0], white[1] + 4), (1, 1, 1), 0.9)
    return canvas


def render_scope_image(
    images: torch.Tensor,
    scope_type: str,
    width: int = 768,
    height: int = 512,
    intensity: float = 1.25,
    show_grid: bool = True,
    analysis_resolution: int = 512,
) -> torch.Tensor:
    outputs = []
    for image in images:
        frame = _analysis_image(image, analysis_resolution)
        if scope_type == SCOPE_TYPES[0]:
            scope = _waveform(frame, height, width, intensity, show_grid, parade=True)
        elif scope_type == SCOPE_TYPES[1]:
            scope = _waveform(frame, height, width, intensity, show_grid, parade=False)
        elif scope_type == SCOPE_TYPES[2]:
            scope = _vectorscope(frame, height, width, intensity, show_grid)
        elif scope_type == SCOPE_TYPES[3]:
            scope = _histogram(frame, height, width, intensity, show_grid)
        elif scope_type == SCOPE_TYPES[4]:
            scope = _cie_scope(frame, height, width, intensity, show_grid)
        else:
            raise ValueError(f"Unknown scope_type: {scope_type}")
        outputs.append(scope.clamp(0.0, 1.0))
    return torch.stack(outputs, dim=0)


class KKDaVinciScopes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "scope_type": (SCOPE_TYPES, {"default": SCOPE_TYPES[1]}),
                "width": ("INT", {"default": 768, "min": 320, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1536, "step": 8}),
                "intensity": ("FLOAT", {"default": 1.25, "min": 0.1, "max": 5.0, "step": 0.05}),
                "show_grid": ("BOOLEAN", {"default": True}),
                "analysis_resolution": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "image/color"
    DESCRIPTION = "DaVinci Resolve inspired RGB parade, waveform, vectorscope, histogram and CIE scope."

    def __init__(self):
        self._preview = PreviewImage()

    def render(
        self,
        images,
        scope_type,
        width,
        height,
        intensity,
        show_grid,
        analysis_resolution,
        prompt=None,
        extra_pnginfo=None,
    ):
        scope_images = render_scope_image(
            images,
            scope_type,
            width,
            height,
            intensity,
            show_grid,
            analysis_resolution,
        )
        preview = self._preview.save_images(
            scope_images,
            filename_prefix="KK_scope",
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        descriptors = preview["ui"]["images"]
        for index, descriptor in enumerate(descriptors):
            descriptor["scope_type"] = scope_type
            descriptor["frame_index"] = index
            descriptor["frame_count"] = len(descriptors)
            descriptor["scope_width"] = width
            descriptor["scope_height"] = height
        # Use a custom UI key so ComfyUI does not create its normal image
        # preview. The plugin's browser extension paints this inside the node.
        return {"ui": {"kk_scope": descriptors}}
