"""Differentiable, interpretable automatic color grading for KK scopes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import kornia
import torch
import torch.nn.functional as F


PARAMETER_NAMES = (
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "temperature",
    "tint",
    "vibrance",
    "saturation",
)

_PARAMETER_RANGES = (2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
_DISPLAY_SCALES = (1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0)


@dataclass
class AutoGradeResult:
    image: torch.Tensor
    parameters: dict[str, float]
    before_error: float
    after_error: float
    improvement: float

    def parameters_json(self) -> str:
        payload = {
            "engine": "KK Auto Grade v1",
            **self.parameters,
            "before_error": round(self.before_error, 6),
            "after_error": round(self.after_error, 6),
            "match_improvement": round(self.improvement, 3),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055).pow(2.4),
    )


def _linear_to_srgb_unclipped(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(0.0)
    return torch.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * value.pow(1.0 / 2.4) - 0.055,
    )


def _smoothstep(low: float, high: float, value: torch.Tensor) -> torch.Tensor:
    t = ((value - low) / (high - low)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _tone_adjust(luma: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    contrast, highlights, shadows, whites, blacks = parameters[1:6]
    value = luma.clamp(1e-4, 1.0 - 1e-4)
    contrast_factor = torch.pow(value.new_tensor(2.0), contrast * 1.35)
    value = torch.sigmoid(torch.logit(value) * contrast_factor)

    black_weight = 1.0 - _smoothstep(0.05, 0.30, value)
    shadow_weight = (1.0 - _smoothstep(0.30, 0.62, value)) * _smoothstep(0.01, 0.30, value)
    highlight_weight = _smoothstep(0.38, 0.72, value) * (1.0 - _smoothstep(0.72, 0.995, value))
    white_weight = _smoothstep(0.70, 0.97, value)
    value = value + blacks * 0.12 * black_weight
    value = value + shadows * 0.18 * shadow_weight
    value = value + highlights * 0.18 * highlight_weight
    value = value + whites * 0.12 * white_weight
    return value.clamp(0.0, 1.0)


def decode_parameters(raw_parameters: torch.Tensor) -> torch.Tensor:
    ranges = raw_parameters.new_tensor(_PARAMETER_RANGES)
    return torch.tanh(raw_parameters) * ranges


def parameter_dict(parameters: torch.Tensor) -> dict[str, float]:
    values = parameters.detach().float().cpu().tolist()
    result = {}
    for name, value, scale in zip(PARAMETER_NAMES, values, _DISPLAY_SCALES):
        result[name] = round(float(value) * scale, 4)
    return result


def apply_auto_grade_parameters(image: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply bounded PS-style parameters to a BHWC sRGB tensor."""
    rgb = image.permute(0, 3, 1, 2).float().clamp(0.0, 1.0)
    exposure = parameters[0]
    exposed_linear = _srgb_to_linear(rgb) * torch.pow(rgb.new_tensor(2.0), exposure)
    exposed_srgb = _linear_to_srgb_unclipped(exposed_linear)
    exposure_clip = F.relu(exposed_srgb - 1.0).mean()
    exposed_srgb = exposed_srgb.clamp(0.0, 1.0)

    lab = kornia.color.rgb_to_lab(exposed_srgb)
    normalized_luma = lab[:, 0:1] / 100.0
    lab[:, 0:1] = _tone_adjust(normalized_luma, parameters) * 100.0

    temperature, tint, vibrance, saturation = parameters[6:10]
    a = lab[:, 1:2] + tint * 12.0
    b = lab[:, 2:3] + temperature * 18.0
    chroma = torch.sqrt(a.square() + b.square() + 1e-6)
    chroma_normalized = (chroma / 80.0).clamp(0.0, 1.0)
    saturation_factor = torch.pow(lab.new_tensor(2.0), saturation * 0.80)
    vibrance_factor = 1.0 + vibrance * 0.75 * (1.0 - chroma_normalized)
    chroma_factor = (saturation_factor * vibrance_factor).clamp(0.20, 2.50)
    lab[:, 1:2] = a * chroma_factor
    lab[:, 2:3] = b * chroma_factor

    unbounded = kornia.color.lab_to_rgb(lab, clip=False)
    gamut_penalty = (F.relu(-unbounded) + F.relu(unbounded - 1.0)).mean()
    output = unbounded.clamp(0.0, 1.0).permute(0, 2, 3, 1)
    return output, exposure_clip + gamut_penalty


def _sample_analysis(image: torch.Tensor, resolution: int, max_frames: int = 4, max_pixels: int = 8192) -> torch.Tensor:
    batch, height, width, _ = image.shape
    if batch > max_frames:
        frame_indices = torch.linspace(0, batch - 1, max_frames, device=image.device).round().long()
        image = image.index_select(0, frame_indices)
    max_side = max(height, width)
    scale = min(1.0, float(resolution) / float(max_side))
    target_height = max(24, int(round(height * scale)))
    target_width = max(24, int(round(width * scale)))
    bchw = image.permute(0, 3, 1, 2).float()
    if (target_height, target_width) != (height, width):
        bchw = F.interpolate(bchw, size=(target_height, target_width), mode="area")
    pixels = bchw.permute(0, 2, 3, 1).reshape(-1, 3)
    if pixels.shape[0] > max_pixels:
        indices = torch.linspace(0, pixels.shape[0] - 1, max_pixels, device=image.device).round().long()
        pixels = pixels.index_select(0, indices)
    return pixels


def _distribution_features(pixels: torch.Tensor) -> torch.Tensor:
    rgb = pixels.clamp(0.0, 1.0)
    luma = rgb[:, 0:1] * 0.2126 + rgb[:, 1:2] * 0.7152 + rgb[:, 2:3] * 0.0722
    saturation = rgb.max(dim=1, keepdim=True).values - rgb.min(dim=1, keepdim=True).values
    lab = kornia.color.rgb_to_lab(rgb.t().unsqueeze(0).unsqueeze(-1)).squeeze(0).squeeze(-1).t()
    a = ((lab[:, 1:2] + 128.0) / 255.0).clamp(0.0, 1.0)
    b = ((lab[:, 2:3] + 128.0) / 255.0).clamp(0.0, 1.0)
    return torch.cat((luma, rgb, a, b, saturation), dim=1)


def _soft_cdf(features: torch.Tensor, bins: int = 48) -> torch.Tensor:
    centers = torch.linspace(0.0, 1.0, bins, device=features.device, dtype=features.dtype)
    sigma = 1.35 / float(bins - 1)
    distances = (features.unsqueeze(-1) - centers.view(1, 1, -1)) / sigma
    histogram = torch.exp(-0.5 * distances.square()).sum(dim=0)
    histogram = histogram / histogram.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return histogram.cumsum(dim=-1)


def _distribution_error(reference_cdf: torch.Tensor, target_pixels: torch.Tensor) -> torch.Tensor:
    target_cdf = _soft_cdf(_distribution_features(target_pixels))
    feature_weights = target_cdf.new_tensor([2.2, 1.0, 1.0, 1.0, 0.65, 0.65, 0.55]).view(-1, 1)
    return ((target_cdf - reference_cdf).abs() * feature_weights).mean()


def _initial_raw_parameters(reference_pixels: torch.Tensor, target_pixels: torch.Tensor) -> torch.Tensor:
    reference_luma = (reference_pixels * reference_pixels.new_tensor([0.2126, 0.7152, 0.0722])).sum(dim=1)
    target_luma = (target_pixels * target_pixels.new_tensor([0.2126, 0.7152, 0.0722])).sum(dim=1)
    reference_mid = torch.quantile(reference_luma, 0.50).clamp_min(0.01)
    target_mid = torch.quantile(target_luma, 0.50).clamp_min(0.01)
    exposure = torch.log2(reference_mid / target_mid).clamp(-1.75, 1.75)
    normalized = (exposure / _PARAMETER_RANGES[0]).clamp(-0.95, 0.95)
    raw = torch.zeros(len(PARAMETER_NAMES), device=target_pixels.device, dtype=target_pixels.dtype)
    raw[0] = torch.atanh(normalized)
    return raw


def _monotonic_penalty(parameters: torch.Tensor) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, 129, device=parameters.device, dtype=parameters.dtype).view(1, 1, 1, -1)
    curve = _tone_adjust(grid, parameters).flatten()
    increments = curve[1:] - curve[:-1]
    return F.relu(1e-4 - increments).mean()


def auto_grade_to_reference(
    reference: torch.Tensor,
    target: torch.Tensor,
    strength: float = 1.0,
    analysis_resolution: int = 192,
    steps: int = 56,
) -> AutoGradeResult:
    """Solve a shared interpretable grade that moves target distributions toward reference."""
    if strength <= 0.0:
        zeros = target.new_zeros(len(PARAMETER_NAMES))
        return AutoGradeResult(target, parameter_dict(zeros), 0.0, 0.0, 0.0)

    solve_resolution = min(max(int(analysis_resolution), 64), 192)
    with torch.inference_mode(False), torch.enable_grad():
        reference_work = reference.detach().float().clone()
        target_work = target.detach().float().clone()
        reference_pixels = _sample_analysis(reference_work, solve_resolution).detach()
        target_pixels = _sample_analysis(target_work, solve_resolution).detach()
        reference_cdf = _soft_cdf(_distribution_features(reference_pixels)).detach()
        before_error_tensor = _distribution_error(reference_cdf, target_pixels).detach()

        raw = torch.nn.Parameter(_initial_raw_parameters(reference_pixels, target_pixels))
        optimizer = torch.optim.Adam([raw], lr=0.075)
        best_error = float("inf")
        best_parameters = decode_parameters(raw).detach().clone()
        pixel_batch = target_pixels.view(1, 1, -1, 3)

        for step in range(max(8, int(steps))):
            optimizer.zero_grad(set_to_none=True)
            parameters = decode_parameters(raw)
            graded_pixels, clipping = apply_auto_grade_parameters(pixel_batch, parameters)
            distribution = _distribution_error(reference_cdf, graded_pixels.reshape(-1, 3))
            regularization = (parameters / parameters.new_tensor(_PARAMETER_RANGES)).square().mean()
            loss = distribution + clipping * 0.35 + _monotonic_penalty(parameters) * 1.5 + regularization * 0.0015
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], 5.0)
            optimizer.step()
            error_value = float(distribution.detach())
            if error_value < best_error:
                best_error = error_value
                best_parameters = parameters.detach().clone()
            if step == int(steps * 0.70):
                for group in optimizer.param_groups:
                    group["lr"] = 0.025

        applied_parameters = best_parameters * float(strength)
        with torch.no_grad():
            adjusted, _ = apply_auto_grade_parameters(target_work, applied_parameters)
            adjusted_pixels = _sample_analysis(adjusted, solve_resolution)
            after_error = float(_distribution_error(reference_cdf, adjusted_pixels))

    before_error = float(before_error_tensor)
    if before_error <= 1e-7:
        improvement = 100.0
    else:
        improvement = max(-999.0, min(100.0, (1.0 - after_error / before_error) * 100.0))
    return AutoGradeResult(
        image=adjusted.detach(),
        parameters=parameter_dict(applied_parameters),
        before_error=before_error,
        after_error=after_error,
        improvement=improvement,
    )

