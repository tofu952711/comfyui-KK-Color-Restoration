"""Two-image color comparison, primary correction, and embedded scopes."""

from __future__ import annotations

import math

import torch

import comfy.model_management
from nodes import PreviewImage

from .auto_grade import PARAMETER_NAMES, AutoGradeResult, auto_grade_to_reference
from .scopes import SCOPE_TYPES, _analysis_image, render_scope_image


COMPARE_VIEWS = (
    "左右对比 (Side by Side)",
    "叠加对比 (Overlay)",
    "差异对比 (Difference)",
)


def _srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055).pow(2.4),
    )


def _linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(0.0)
    return torch.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * value.pow(1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)


def _repeat_reference(reference: torch.Tensor, batch: int) -> torch.Tensor:
    indices = torch.arange(batch, device=reference.device) % reference.shape[0]
    return reference.index_select(0, indices)


def apply_primary_adjustments(
    image: torch.Tensor,
    exposure_ev: float = 0.0,
    temperature: float = 0.0,
    tint: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> torch.Tensor:
    """Apply compact DaVinci-style primary numeric adjustments to image 2."""
    linear = _srgb_to_linear(image.float())
    temp = float(temperature) / 100.0
    tint_value = float(tint) / 100.0
    gains = linear.new_tensor(
        [
            1.0 + 0.20 * temp + 0.05 * tint_value,
            1.0 - 0.10 * tint_value,
            1.0 - 0.20 * temp + 0.05 * tint_value,
        ]
    ).clamp(0.5, 1.5)
    linear = linear * gains * (2.0 ** float(exposure_ev))
    rgb = _linear_to_srgb(linear)
    rgb = (rgb - 0.5) * float(contrast) + 0.5
    luma = (
        rgb[..., 0:1] * 0.2126
        + rgb[..., 1:2] * 0.7152
        + rgb[..., 2:3] * 0.0722
    )
    return torch.lerp(luma, rgb, float(saturation)).clamp(0.0, 1.0)


def _frame_metrics(image: torch.Tensor, analysis_resolution: int) -> dict:
    frame = _analysis_image(image, analysis_resolution)
    luma = frame[..., 0] * 0.2126 + frame[..., 1] * 0.7152 + frame[..., 2] * 0.0722
    saturation = frame.max(dim=-1).values - frame.min(dim=-1).values
    rgb_mean = frame.mean(dim=(0, 1))
    return {
        "luma": round(float(luma.mean()) * 100.0, 3),
        "contrast": round(float(luma.std(unbiased=False)) * 100.0, 3),
        "saturation": round(float(saturation.mean()) * 100.0, 3),
        "red": round(float(rgb_mean[0]) * 100.0, 3),
        "green": round(float(rgb_mean[1]) * 100.0, 3),
        "blue": round(float(rgb_mean[2]) * 100.0, 3),
    }


def _metric_comparison(reference: dict, target: dict) -> dict:
    delta = {key: round(target[key] - reference[key], 3) for key in reference}
    ref_luma = max(reference["luma"], 0.01)
    target_luma = max(target["luma"], 0.01)
    ref_contrast = max(reference["contrast"], 0.01)
    target_contrast = max(target["contrast"], 0.01)
    ref_saturation = max(reference["saturation"], 0.01)
    target_saturation = max(target["saturation"], 0.01)
    suggestions = {
        "exposure_ev": round(math.log2(ref_luma / target_luma), 3),
        "contrast": round(ref_contrast / target_contrast, 3),
        "saturation": round(ref_saturation / target_saturation, 3),
    }
    return {"reference": reference, "target": target, "delta": delta, "suggestions": suggestions}


class KKDualImageColorScopes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE", {"tooltip": "参考图1：只用于提供目标颜色数据，不会被修改。"}),
                "target_image": ("IMAGE", {"tooltip": "输入图2：自动匹配和手动参数都会实际修改这张图的像素。"}),
                "scope_type": (SCOPE_TYPES, {"default": SCOPE_TYPES[1], "tooltip": "仅改变示波器类型，不修改图片。"}),
                "comparison_view": (COMPARE_VIEWS, {"default": COMPARE_VIEWS[0], "tooltip": "仅改变节点内部对比显示，不修改图片。"}),
                "auto_match": ("BOOLEAN", {"default": True, "tooltip": "开启自动调色求解器：自动操作曝光、明暗分区、白平衡和饱和度，使图2示波器接近图1。"}),
                "match_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "自动求解参数应用强度；1.0 为完整自动调色。"}),
                "exposure_ev": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.01, "tooltip": "自动调色完成后的曝光微调，0 为不追加。"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "自动调色完成后的色温微调，0 为不追加。"}),
                "tint": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "自动调色完成后的绿/洋红微调，0 为不追加。"}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "自动调色完成后的对比度倍率，1.0 为不追加。"}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "自动调色完成后的饱和度倍率，1.0 为不追加。"}),
                "analysis_resolution": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64, "tooltip": "示波器分析分辨率；自动求解内部会限制采样规模以保证速度。"}),
                "difference_gain": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 12.0, "step": 0.25, "tooltip": "只放大差异示波器的显示，不修改图2。"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("image2_adjusted", "grade_parameters", "match_score")
    FUNCTION = "compare_and_adjust"
    OUTPUT_NODE = True
    CATEGORY = "image/color"
    DESCRIPTION = "Automatically solve interpretable grading controls so image 2 distributions and scopes approach reference image 1."

    def __init__(self):
        self._preview = PreviewImage()

    def compare_and_adjust(
        self,
        reference_image,
        target_image,
        scope_type,
        comparison_view,
        auto_match,
        match_strength,
        exposure_ev,
        temperature,
        tint,
        contrast,
        saturation,
        analysis_resolution,
        difference_gain=4.0,
        prompt=None,
        extra_pnginfo=None,
    ):
        device = target_image.device
        reference = reference_image.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        target = target_image.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        if auto_match:
            grade = auto_grade_to_reference(
                reference,
                target,
                strength=match_strength,
                analysis_resolution=analysis_resolution,
            )
        else:
            grade = AutoGradeResult(
                image=target,
                parameters={name: 0.0 for name in PARAMETER_NAMES},
                before_error=0.0,
                after_error=0.0,
                improvement=0.0,
            )
        adjusted = grade.image
        adjusted = apply_primary_adjustments(
            adjusted,
            exposure_ev=exposure_ev,
            temperature=temperature,
            tint=tint,
            contrast=contrast,
            saturation=saturation,
        )

        reference_batch = _repeat_reference(reference, adjusted.shape[0])
        scope_reference = render_scope_image(
            reference_batch, scope_type, 720, 420, 1.25, True, analysis_resolution
        )
        scope_target = render_scope_image(
            adjusted, scope_type, 720, 420, 1.25, True, analysis_resolution
        )
        reference_only = (scope_reference - scope_target).clamp_min(0.0).mean(dim=-1, keepdim=True)
        target_only = (scope_target - scope_reference).clamp_min(0.0).mean(dim=-1, keepdim=True)
        reference_color = scope_reference.new_tensor([0.10, 0.72, 1.00]).view(1, 1, 1, 3)
        target_color = scope_reference.new_tensor([1.00, 0.42, 0.08]).view(1, 1, 1, 3)
        scope_difference = (
            reference_only * reference_color + target_only * target_color
        ) * float(difference_gain)
        scope_difference = scope_difference.clamp(0.0, 1.0)
        preview = self._preview.save_images(
            torch.cat((scope_reference, scope_target, scope_difference), dim=0),
            filename_prefix="KK_compare_scope",
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        descriptors = preview["ui"]["images"]
        batch = adjusted.shape[0]
        pairs = []
        for index in range(batch):
            reference_metrics = _frame_metrics(reference_batch[index], analysis_resolution)
            target_metrics = _frame_metrics(adjusted[index], analysis_resolution)
            pixel_delta = (adjusted[index].float() - target[index].float()).abs()
            comparison = _metric_comparison(reference_metrics, target_metrics)
            comparison["adjustment"] = {
                "auto_match": bool(auto_match),
                "mean_pixel_change": round(float(pixel_delta.mean()) * 100.0, 4),
                "max_pixel_change": round(float(pixel_delta.max()) * 100.0, 4),
                "parameters": grade.parameters,
                "before_error": round(grade.before_error, 6),
                "after_error": round(grade.after_error, 6),
                "match_score": round(grade.improvement, 3),
            }
            pairs.append(
                {
                    "reference": descriptors[index],
                    "target": descriptors[batch + index],
                    "difference": descriptors[batch * 2 + index],
                    "metrics": comparison,
                    "scope_type": scope_type,
                    "comparison_view": comparison_view,
                    "frame_index": index,
                    "frame_count": batch,
                }
            )

        intermediate_device = comfy.model_management.intermediate_device()
        intermediate_dtype = comfy.model_management.intermediate_dtype()
        return {
            "ui": {"kk_compare_scope": pairs},
            "result": (
                adjusted.to(device=intermediate_device, dtype=intermediate_dtype),
                grade.parameters_json(),
                float(grade.improvement),
            ),
        }
