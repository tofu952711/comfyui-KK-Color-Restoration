"""ComfyUI ColorTransfer with optional, invertible mask compositing.

The color transforms intentionally follow ComfyUI's built-in ColorTransfer
node. Images are BHWC RGB float tensors. Masks are BHW float tensors.
"""

from __future__ import annotations

import kornia
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils


METHODS = ("reinhard_lab", "mkl_lab", "histogram")
SOURCE_STATS = ("per_frame", "uniform", "target_frame")


def _prepare_mask(mask: torch.Tensor | None, batch: int, height: int, width: int, device) -> torch.Tensor:
    """Normalize common ComfyUI MASK layouts to BHW on the working device."""
    if mask is None:
        return torch.ones((batch, height, width), device=device, dtype=torch.float32)

    mask = mask.to(device=device, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4:
        # Accept BHWC/BCHW masks produced by a few third-party segmentation nodes.
        if mask.shape[-1] in (1, 3, 4):
            mask = mask[..., 0]
        elif mask.shape[1] in (1, 3, 4):
            mask = mask[:, 0]
        else:
            raise ValueError(f"Unsupported 4D mask shape: {tuple(mask.shape)}")
    if mask.ndim != 3:
        raise ValueError(f"MASK must be HW, BHW, BHWC, or BCHW; got {tuple(mask.shape)}")

    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)

    if mask.shape[0] != batch:
        # Match ComfyUI's common batch behavior: cycle the available masks.
        indices = torch.arange(batch, device=device) % mask.shape[0]
        mask = mask.index_select(0, indices)
    return mask.clamp(0.0, 1.0)


def _gaussian_feather(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    sigma = max(radius / 3.0, 0.5)
    return kornia.filters.gaussian_blur2d(
        mask.unsqueeze(1),
        (kernel_size, kernel_size),
        (sigma, sigma),
        border_type="replicate",
    ).squeeze(1)


class KKColorRestore:
    """Built-in ColorTransfer behavior plus an optional output mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE",),
                "image_ref": ("IMAGE",),
                "method": (METHODS, {"default": "reinhard_lab"}),
                "source_stats": (SOURCE_STATS, {"default": "per_frame"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "mask_feather": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "target_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
                "reference_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "effective_mask")
    FUNCTION = "transfer"
    CATEGORY = "image/color"
    DESCRIPTION = (
        "ComfyUI ColorTransfer with an optional mask. Unconnected mask means full image. "
        "invert_mask swaps the adjusted and protected regions."
    )

    @staticmethod
    def _to_lab(images: torch.Tensor, index: int, device) -> torch.Tensor:
        return kornia.color.rgb_to_lab(
            images[index : index + 1].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        )

    @staticmethod
    def _frame_weights(mask: torch.Tensor | None, index: int, pixels: int, device) -> torch.Tensor | None:
        if mask is None:
            return None
        weights = mask[index].to(device=device, dtype=torch.float32).reshape(1, -1).clamp(0.0, 1.0)
        # Tiny/empty segmentation results are not statistically meaningful.
        if float(weights.sum()) < 16.0:
            return torch.ones((1, pixels), device=device, dtype=torch.float32)
        return weights

    @classmethod
    def _pool_stats(
        cls,
        images: torch.Tensor,
        device,
        is_reinhard: bool,
        eps: float,
        masks: torch.Tensor | None = None,
    ):
        frames, channels = images.shape[0], images.shape[3]
        pixels = images.shape[1] * images.shape[2]
        mean = torch.zeros(channels, 1, device=device, dtype=torch.float32)
        for i in range(frames):
            lab = cls._to_lab(images, i, device).view(channels, -1)
            weights = cls._frame_weights(masks, i, pixels, device)
            if weights is None:
                mean += lab.mean(dim=-1, keepdim=True)
            else:
                mean += (lab * weights).sum(dim=-1, keepdim=True) / weights.sum()
        mean /= frames

        acc = torch.zeros(channels, 1 if is_reinhard else channels, device=device, dtype=torch.float32)
        for i in range(frames):
            centered = cls._to_lab(images, i, device).view(channels, -1) - mean
            weights = cls._frame_weights(masks, i, pixels, device)
            if is_reinhard:
                if weights is None:
                    acc += (centered * centered).mean(dim=-1, keepdim=True)
                else:
                    acc += (centered.square() * weights).sum(dim=-1, keepdim=True) / weights.sum()
            else:
                if weights is None:
                    acc += centered @ centered.T / pixels
                else:
                    acc += (centered * weights) @ centered.T / weights.sum()
        if is_reinhard:
            return mean, torch.sqrt(acc / frames).clamp_min_(eps)
        return mean, acc / frames

    @staticmethod
    def _frame_stats(
        lab_flat: torch.Tensor,
        pixels: int,
        is_reinhard: bool,
        eps: float,
        weights: torch.Tensor | None = None,
    ):
        if weights is None:
            mean = lab_flat.mean(dim=-1, keepdim=True)
        else:
            mean = (lab_flat * weights).sum(dim=-1, keepdim=True) / weights.sum()
        if is_reinhard:
            if weights is None:
                scale = lab_flat.std(dim=-1, keepdim=True, unbiased=False)
            else:
                scale = torch.sqrt(
                    ((lab_flat - mean).square() * weights).sum(dim=-1, keepdim=True) / weights.sum()
                )
            return mean, scale.clamp_min_(eps)
        centered = lab_flat - mean
        covariance = centered @ centered.T / pixels if weights is None else (centered * weights) @ centered.T / weights.sum()
        return mean, covariance

    @staticmethod
    def _mkl_matrix(cov_source: torch.Tensor, cov_ref: torch.Tensor, eps: float) -> torch.Tensor:
        eig_val_s, eig_vec_s = torch.linalg.eigh(cov_source)
        sqrt_val_s = torch.sqrt(eig_val_s.clamp_min(0)).clamp_min_(eps)

        scaled_vectors = eig_vec_s * sqrt_val_s.unsqueeze(0)
        middle = scaled_vectors.T @ cov_ref @ scaled_vectors
        eig_val_m, eig_vec_m = torch.linalg.eigh(middle)
        sqrt_middle = torch.sqrt(eig_val_m.clamp_min(0))

        inverse_scaled_vectors = eig_vec_s * (1.0 / sqrt_val_s).unsqueeze(0)
        middle_half = (eig_vec_m * sqrt_middle.unsqueeze(0)) @ eig_vec_m.T
        return inverse_scaled_vectors @ middle_half @ inverse_scaled_vectors.T

    @staticmethod
    def _histogram_lut(
        source: torch.Tensor,
        reference: torch.Tensor,
        bins: int = 256,
        source_weights: torch.Tensor | None = None,
        reference_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_bins = (source * (bins - 1)).long().clamp(0, bins - 1)
        reference_bins = (reference * (bins - 1)).long().clamp(0, bins - 1)
        source_hist = torch.zeros(source.shape[0], bins, device=source.device, dtype=source.dtype)
        reference_hist = torch.zeros_like(source_hist)
        source_values = torch.ones_like(source) if source_weights is None else source_weights.expand_as(source)
        reference_values = (
            torch.ones_like(reference) if reference_weights is None else reference_weights.expand_as(reference)
        )
        source_hist.scatter_add_(1, source_bins, source_values)
        reference_hist.scatter_add_(1, reference_bins, reference_values)
        source_cdf = source_hist.cumsum(1)
        source_cdf = source_cdf / source_cdf[:, -1:]
        reference_cdf = reference_hist.cumsum(1)
        reference_cdf = reference_cdf / reference_cdf[:, -1:]
        return torch.searchsorted(reference_cdf, source_cdf).clamp_max_(bins - 1).float() / (bins - 1)

    @classmethod
    def _pooled_cdf(
        cls,
        images: torch.Tensor,
        device,
        bins: int = 256,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        channels = images.shape[3]
        pixels = images.shape[1] * images.shape[2]
        histogram = torch.zeros(channels, bins, device=device, dtype=torch.float32)
        for i in range(images.shape[0]):
            frame = images[i].to(device=device, dtype=torch.float32).permute(2, 0, 1).reshape(channels, -1)
            frame_bins = (frame * (bins - 1)).long().clamp(0, bins - 1)
            weights = cls._frame_weights(masks, i, pixels, device)
            values = torch.ones_like(frame) if weights is None else weights.expand_as(frame)
            histogram.scatter_add_(1, frame_bins, values)
        cdf = histogram.cumsum(1)
        return cdf / cdf[:, -1:]

    @classmethod
    def _build_histogram_transform(
        cls,
        image_target,
        image_ref,
        device,
        stats_mode: str,
        target_index: int,
        batch: int,
        target_masks: torch.Tensor | None = None,
        reference_masks: torch.Tensor | None = None,
    ):
        if stats_mode == "per_frame":
            return None
        reference_cdf = cls._pooled_cdf(image_ref, device, masks=reference_masks)
        if stats_mode == "target_frame":
            index = min(max(int(target_index), 0), batch - 1)
            selected_mask = None if target_masks is None else target_masks[index : index + 1]
            source_cdf = cls._pooled_cdf(image_target[index : index + 1], device, masks=selected_mask)
        else:
            source_cdf = cls._pooled_cdf(image_target, device, masks=target_masks)
        return torch.searchsorted(reference_cdf, source_cdf).clamp_max_(255).float() / 255.0

    @classmethod
    def _build_lab_transform(
        cls,
        image_target,
        image_ref,
        device,
        stats_mode: str,
        target_index: int,
        is_reinhard: bool,
        target_masks: torch.Tensor | None = None,
        reference_masks: torch.Tensor | None = None,
    ):
        eps = 1e-6
        batch, height, width, channels = image_target.shape
        reference_batch = image_ref.shape[0]
        single_reference = reference_batch == 1
        pixels = height * width
        reference_pixels = image_ref.shape[1] * image_ref.shape[2]

        if single_reference or stats_mode in ("uniform", "target_frame"):
            ref_mean, ref_scale = cls._pool_stats(
                image_ref, device, is_reinhard, eps, masks=reference_masks
            )

        if stats_mode in ("uniform", "target_frame"):
            if stats_mode == "target_frame":
                index = min(max(int(target_index), 0), batch - 1)
                source_lab = cls._to_lab(image_target, index, device).view(channels, -1)
                source_weights = cls._frame_weights(target_masks, index, pixels, device)
                source_mean, source_scale = cls._frame_stats(
                    source_lab, pixels, is_reinhard, eps, weights=source_weights
                )
            else:
                source_mean, source_scale = cls._pool_stats(
                    image_target, device, is_reinhard, eps, masks=target_masks
                )

            if is_reinhard:
                scale = ref_scale / source_scale
                offset = ref_mean - scale * source_mean
                return lambda source_flat, **_: source_flat * scale + offset
            transform = cls._mkl_matrix(source_scale, ref_scale, eps)
            offset = ref_mean - transform @ source_mean
            return lambda source_flat, **_: transform @ source_flat + offset

        def per_frame_transform(source_flat: torch.Tensor, frame_index: int):
            source_weights = cls._frame_weights(target_masks, frame_index, pixels, device)
            source_mean, source_scale = cls._frame_stats(
                source_flat, pixels, is_reinhard, eps, weights=source_weights
            )
            if single_reference:
                reference_mean, reference_scale = ref_mean, ref_scale
            else:
                ref_index = min(frame_index, reference_batch - 1)
                ref_lab = cls._to_lab(image_ref, ref_index, device).view(channels, -1)
                ref_weights = cls._frame_weights(reference_masks, ref_index, reference_pixels, device)
                reference_mean, reference_scale = cls._frame_stats(
                    ref_lab, reference_pixels, is_reinhard, eps, weights=ref_weights
                )

            centered = source_flat - source_mean
            if is_reinhard:
                return centered * (reference_scale / source_scale) + reference_mean
            source_covariance = (
                centered @ centered.T / pixels
                if source_weights is None
                else (centered * source_weights) @ centered.T / source_weights.sum()
            )
            transform = cls._mkl_matrix(source_covariance, reference_scale, eps)
            return transform @ centered + reference_mean

        return per_frame_transform

    @classmethod
    def _correct_full_image(
        cls,
        image_target: torch.Tensor,
        image_ref: torch.Tensor,
        method: str,
        source_stats: str,
        strength: float,
        target_index: int,
        device,
        target_masks: torch.Tensor | None = None,
        reference_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, height, width, channels = image_target.shape
        reference_batch = image_ref.shape[0]
        output_frames = []

        if method == "histogram":
            uniform_lut = cls._build_histogram_transform(
                image_target,
                image_ref,
                device,
                source_stats,
                target_index,
                batch,
                target_masks=target_masks,
                reference_masks=reference_masks,
            )
            for i in range(batch):
                source = image_target[i].to(device=device, dtype=torch.float32).permute(2, 0, 1)
                source_flat = source.reshape(channels, -1)
                if uniform_lut is None:
                    ref_index = min(i, reference_batch - 1)
                    reference = image_ref[ref_index].to(device=device, dtype=torch.float32)
                    reference = reference.permute(2, 0, 1).reshape(channels, -1)
                    source_weights = cls._frame_weights(
                        target_masks, i, height * width, device
                    )
                    reference_weights = cls._frame_weights(
                        reference_masks,
                        ref_index,
                        image_ref.shape[1] * image_ref.shape[2],
                        device,
                    )
                    lut = cls._histogram_lut(
                        source_flat,
                        reference,
                        source_weights=source_weights,
                        reference_weights=reference_weights,
                    )
                else:
                    lut = uniform_lut
                bin_index = (source_flat * 255).long().clamp(0, 255)
                matched = lut.gather(1, bin_index).view(channels, height, width)
                result = matched if strength == 1.0 else torch.lerp(source, matched, strength)
                output_frames.append(result.permute(1, 2, 0).clamp(0.0, 1.0))
        else:
            transform = cls._build_lab_transform(
                image_target,
                image_ref,
                device,
                source_stats,
                target_index,
                is_reinhard=method == "reinhard_lab",
                target_masks=target_masks,
                reference_masks=reference_masks,
            )
            for i in range(batch):
                source_lab = cls._to_lab(image_target, i, device)
                corrected = transform(source_lab.view(channels, -1), frame_index=i)
                corrected = corrected.view(1, channels, height, width)
                result_lab = corrected if strength == 1.0 else torch.lerp(source_lab, corrected, strength)
                result = kornia.color.lab_to_rgb(result_lab)
                output_frames.append(result.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0))

        return torch.stack(output_frames, dim=0)

    def transfer(
        self,
        image_target: torch.Tensor,
        image_ref: torch.Tensor,
        method: str,
        source_stats: str,
        strength: float,
        invert_mask: bool,
        mask_feather: int,
        target_index: int,
        mask: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
    ):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        if source_stats not in SOURCE_STATS:
            raise ValueError(f"Unknown source_stats: {source_stats}")

        device = comfy.model_management.get_torch_device()
        batch, height, width, _ = image_target.shape
        effective_mask = _prepare_mask(mask, batch, height, width, device)
        if invert_mask and mask is not None:
            effective_mask = 1.0 - effective_mask
        effective_mask = _gaussian_feather(effective_mask, mask_feather).clamp(0.0, 1.0)

        # A connected semantic mask defines both where the correction is
        # applied and which pixels are used to estimate the color transform.
        target_stats_mask = effective_mask if mask is not None else None
        reference_stats_mask = None
        if reference_mask is not None or mask is not None:
            reference_source = reference_mask if reference_mask is not None else mask
            reference_stats_mask = _prepare_mask(
                reference_source,
                image_ref.shape[0],
                image_ref.shape[1],
                image_ref.shape[2],
                device,
            )
            if invert_mask:
                reference_stats_mask = 1.0 - reference_stats_mask
            reference_stats_mask = _gaussian_feather(
                reference_stats_mask, mask_feather
            ).clamp(0.0, 1.0)

        # Preserve exact parity with the built-in node for full-white masks.
        if target_stats_mask is not None and float(target_stats_mask.min()) >= 0.999999:
            target_stats_mask = None
        if reference_stats_mask is not None and float(reference_stats_mask.min()) >= 0.999999:
            reference_stats_mask = None

        target = image_target.to(device=device, dtype=torch.float32)
        if strength == 0.0 or float(effective_mask.max()) == 0.0:
            result = target
        else:
            corrected = self._correct_full_image(
                image_target,
                image_ref,
                method,
                source_stats,
                strength,
                target_index,
                device,
                target_masks=target_stats_mask,
                reference_masks=reference_stats_mask,
            )
            result = torch.lerp(target, corrected, effective_mask.unsqueeze(-1)).clamp(0.0, 1.0)

        intermediate_device = comfy.model_management.intermediate_device()
        intermediate_dtype = comfy.model_management.intermediate_dtype()
        return (
            result.to(device=intermediate_device, dtype=intermediate_dtype),
            effective_mask.to(device=intermediate_device, dtype=intermediate_dtype),
        )
