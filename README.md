# ComfyUI KK Color Restoration

> 面向 AI 图像编辑工作流的 ComfyUI 色彩还原与自动参考调色工具集。

`ComfyUI KK Color Restoration` 提供遮罩感知的颜色迁移、可微分自动调色，以及嵌入节点内部的专业示波器。它主要用于修复 AI 重绘、局部编辑或整图重新采样后产生的全局偏色，并支持通过语义分割遮罩限定统计区域和应用区域。

**English:** A mask-aware color restoration and automatic reference grading toolkit for ComfyUI, featuring differentiable PS-style controls and embedded DaVinci-inspired scopes.

## 主要功能

- 遮罩感知色彩还原：支持目标遮罩、参考遮罩、反转遮罩和羽化。
- 三种颜色迁移方法：Reinhard Lab、MKL Lab 和直方图匹配。
- 自动参考调色：自动求解曝光、对比度、高光、阴影、白色、黑色、色温、色调、自然饱和度和饱和度。
- 可解释参数输出：自动调色结果可输出为 JSON，并提供匹配改善分数。
- 专业视频示波器：RGB Parade、RGB Waveform、Vectorscope、Histogram 和 CIE 1931。
- 双图动态监看：支持左右、叠加和增强差异视图。
- 批次支持：可处理 ComfyUI `IMAGE` 批次输入。
- 无额外模型下载：自动调色基于 PyTorch/Kornia 优化，不依赖生成模型。

## 节点

所有节点位于 `image/color` 分类。

### KK色彩还原

对编辑后的目标图执行颜色迁移，并通过可选遮罩限制颜色统计和应用区域。

主要输入：

- `image_target`：需要还原颜色的 AI 编辑图。
- `image_ref`：编辑前的参考图。
- `method`：`reinhard_lab`、`mkl_lab` 或 `histogram`。
- `source_stats`：`per_frame`、`uniform` 或 `target_frame`。
- `strength`：颜色迁移强度。
- `mask`：目标图遮罩；白色区域参与统计并应用调色。
- `reference_mask`：参考图遮罩；适合两张图位置或姿态不完全一致的情况。
- `invert_mask`：反转遮罩。
- `mask_feather`：遮罩边缘羽化半径。

输出：

- `image`：色彩还原后的图像。
- `effective_mask`：完成缩放、反转和羽化后的实际遮罩。

### KK双图调色示波器

使用 `KK Auto Grade v1` 自动寻找一组可解释的调色参数，使图2的亮度、RGB、Lab 色度与饱和度分布接近参考图1。该节点不是 Reinhard 颜色迁移的重复实现。

自动求解参数：

- Exposure / 曝光
- Contrast / 对比度
- Highlights / 高光
- Shadows / 阴影
- Whites / 白色
- Blacks / 黑色
- Temperature / 色温
- Tint / 色调
- Vibrance / 自然饱和度
- Saturation / 饱和度

输出：

- `image2_adjusted`：自动调色后的图2。
- `grade_parameters`：自动求解参数 JSON。
- `match_score`：调整前后分布误差的改善百分比。

节点内部显示参考示波器、调整后示波器、增强差异图、自动调色参数、实际像素变化和匹配误差。

> `target_image` 应连接未经 `KK色彩还原` 处理的原始图2，避免重复调色。

### KK达芬奇示波器

嵌入节点内部的动态监看终点，支持：

- RGB Parade
- RGB Waveform
- Vectorscope
- Histogram
- CIE 1931 色度图与 Rec.709 三角形

示波器只负责分析和监看，不修改输入图像。

## 安装

进入 ComfyUI 的 `custom_nodes` 目录后执行：

```bash
git clone https://github.com/tofu952711/comfyui-KK-Color-Restoration.git comfyui_masked_color_transfer
cd comfyui_masked_color_transfer
pip install -r requirements.txt
```

使用 ComfyUI 便携版时，请用 ComfyUI 自带的 Python 安装依赖。安装完成后重启 ComfyUI，并在浏览器中强制刷新页面。

## 推荐工作流

全图颜色还原：

```text
编辑前原图 ──→ image_ref
AI 编辑图  ──→ image_target
                 KK色彩还原 ──→ 输出
```

局部语义调色：

```text
编辑前原图 ─────────────→ image_ref
AI 编辑图  ─────────────→ image_target
语义分割遮罩 ───────────→ mask
参考图语义分割遮罩 ─────→ reference_mask
                              KK色彩还原 ──→ 输出
```

自动参考调色：

```text
参考图1 ──→ reference_image
原始图2 ──→ target_image
              KK双图调色示波器 ──→ image2_adjusted
```

## 技术架构

- PyTorch：张量处理、GPU 加速和自动求导。
- Kornia：Lab/RGB 色彩空间转换与可微图像处理。
- Adam：自动求解 PS 风格调色参数。
- CDF 分布损失：比较亮度、RGB、Lab 色度与饱和度分布。
- ComfyUI Web Extension：节点内嵌 Canvas 示波器和动态参数面板。

自动调色仅优化受约束的可解释参数，而不是直接生成或任意改写像素。处理完成后，求解参数会应用到全分辨率图像。

## 测试

在 ComfyUI 根目录执行：

```bash
python custom_nodes/comfyui_masked_color_transfer/tests/test_masked_color_transfer.py
python custom_nodes/comfyui_masked_color_transfer/tests/test_scopes.py
python custom_nodes/comfyui_masked_color_transfer/tests/test_compare_scopes.py
```

## 使用建议

- AI 编辑前后构图基本一致时，自动参考调色效果最稳定。
- 两张图主体比例差异较大时，优先使用语义遮罩限定统计区域。
- 视频工作流应避免逐帧使用差异过大的参考图，以减少颜色闪烁。
- 示波器接近不代表必须逐像素相同；节点目标是减少整体颜色和明暗分布差异，同时保持图2内容结构。
