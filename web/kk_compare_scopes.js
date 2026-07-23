const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const COMPARE_STYLE_ID = "kk-dual-scope-style";
const COMPARE_WIDGET_HEIGHT = 640;

function installCompareStyles() {
    if (document.getElementById(COMPARE_STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = COMPARE_STYLE_ID;
    style.textContent = `
        .kk-compare-root {
            width:100%; height:${COMPARE_WIDGET_HEIGHT}px;
            min-height:${COMPARE_WIDGET_HEIGHT}px; max-height:${COMPARE_WIDGET_HEIGHT}px;
            display:flex; flex-direction:column; overflow:hidden;
            box-sizing:border-box; background:#07090b; border:1px solid #30353a;
            color:#c9cfd5; font:11px sans-serif; user-select:none;
        }
        .kk-compare-toolbar {
            flex:0 0 30px; display:flex; align-items:center; gap:8px;
            padding:0 8px; background:#1a1e22; border-bottom:1px solid #30353a;
        }
        .kk-compare-live { color:#68717a; font-size:10px; font-weight:700; letter-spacing:.6px; }
        .kk-compare-live::before {
            content:""; display:inline-block; width:7px; height:7px; margin-right:5px;
            border-radius:50%; background:#555c63; vertical-align:-1px;
        }
        .kk-compare-live.active { color:#63e58e; }
        .kk-compare-live.active::before { background:#34dd70; box-shadow:0 0 8px #34dd70; }
        .kk-compare-title { flex:1; text-align:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .kk-compare-time { min-width:48px; color:#737c85; text-align:right; font-size:10px; }
        .kk-compare-stage { position:relative; flex:1 1 auto; min-height:0; background:#030405; }
        .kk-compare-canvas { display:block; width:100%; height:100%; }
        .kk-compare-status {
            position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
            color:#68717a; pointer-events:none;
        }
        .kk-compare-label {
            position:absolute; top:7px; padding:3px 7px; border-radius:3px;
            background:rgba(8,10,12,.78); font-size:10px; pointer-events:none;
        }
        .kk-compare-label.ref { left:8px; color:#73d3ff; }
        .kk-compare-label.target { right:8px; color:#ffb65c; }
        .kk-compare-metrics {
            flex:0 0 272px; padding:5px 8px; box-sizing:border-box;
            background:#111418; border-top:1px solid #30353a; overflow:hidden;
        }
        .kk-compare-table { display:grid; grid-template-columns:1.3fr repeat(3,1fr); gap:2px 8px; }
        .kk-compare-cell { padding:1px 2px; text-align:right; font-variant-numeric:tabular-nums; }
        .kk-compare-cell.name { text-align:left; color:#9aa3ab; }
        .kk-compare-cell.header { color:#7d8790; border-bottom:1px solid #2b3035; padding-bottom:3px; }
        .kk-compare-cell.ref { color:#73d3ff; }
        .kk-compare-cell.target { color:#ffb65c; }
        .kk-compare-cell.delta.positive { color:#ff8a80; }
        .kk-compare-cell.delta.negative { color:#79d990; }
        .kk-compare-suggestion {
            margin-top:5px; padding-top:5px; border-top:1px solid #2b3035;
            color:#9aa3ab; text-align:center; font-variant-numeric:tabular-nums;
        }
        .kk-auto-grade-title {
            margin-top:5px; padding:5px 2px 3px; border-top:1px solid #2b3035;
            color:#d7dce0; font-weight:700;
        }
        .kk-auto-grade-title span { color:#63e58e; float:right; font-weight:600; }
        .kk-auto-grade-grid {
            display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:3px 7px;
            font-variant-numeric:tabular-nums;
        }
        .kk-auto-grade-item {
            display:flex; justify-content:space-between; gap:5px; padding:2px 4px;
            background:#171b1f; border-radius:2px; color:#929ba3;
        }
        .kk-auto-grade-item b { color:#e0e5e9; font-weight:500; }
        .kk-compare-batch {
            display:none; flex:0 0 26px; align-items:center; justify-content:center;
            gap:10px; background:#15191d; border-top:1px solid #30353a;
        }
        .kk-compare-batch.visible { display:flex; }
        .kk-compare-button {
            width:28px; height:19px; border:1px solid #3b4248; border-radius:3px;
            background:#252b30; color:#d3d8dc; cursor:pointer;
        }
        .kk-compare-button:disabled { opacity:.35; cursor:default; }
    `;
    document.head.appendChild(style);
}

function chainCompareCallback(target, name, callback) {
    const original = target[name];
    target[name] = function (...args) {
        const result = original?.apply(this, args);
        callback.apply(this, args);
        return result;
    };
}

function compareImageUrl(info) {
    const params = new URLSearchParams({
        filename: info.filename,
        subfolder: info.subfolder || "",
        type: info.type || "temp",
        t: String(Date.now()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function drawContained(ctx, image, x, y, width, height, alpha = 1.0) {
    if (!image) return;
    const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
    const drawWidth = image.naturalWidth * scale;
    const drawHeight = image.naturalHeight * scale;
    ctx.globalAlpha = alpha;
    ctx.drawImage(image, x + (width - drawWidth) / 2, y + (height - drawHeight) / 2, drawWidth, drawHeight);
    ctx.globalAlpha = 1.0;
}

function migrateLegacyCompareWidgets(node) {
    // Older workflows ended their widgets_values array with the serialized
    // DOM-widget placeholder (""). When difference_gain was added later,
    // LiteGraph assigned that placeholder to the new FLOAT widget by index.
    // ComfyUI then silently excludes this output node during prompt
    // validation because "" cannot be converted to a float.
    let migrated = false;
    const differenceGain = node.widgets?.find((item) => item.name === "difference_gain");
    if (differenceGain) {
        const value = Number(differenceGain.value);
        if (differenceGain.value === "" || !Number.isFinite(value) || value < 1.0 || value > 12.0) {
            differenceGain.value = 4.0;
            migrated = true;
        }
    }

    // Version-one workflows stored auto_match=false even though matching is
    // the primary purpose of this node. Enable it once during migration; the
    // user can still turn it off afterwards and the marker preserves that
    // explicit choice on subsequent loads.
    node.properties = node.properties || {};
    if (!node.properties.kk_auto_match_migrated_v2) {
        const autoMatch = node.widgets?.find((item) => item.name === "auto_match");
        if (autoMatch) {
            autoMatch.value = true;
            migrated = true;
        }
        node.properties.kk_auto_match_migrated_v2 = true;
    }

    if (migrated) node.setDirtyCanvas?.(true, true);
    return migrated;
}

app.registerExtension({
    name: "KK.DualImageColorScopes",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "KKDualImageColorScopes") return;

        chainCompareCallback(nodeType.prototype, "onNodeCreated", function () {
            installCompareStyles();
            const node = this;

            const root = document.createElement("div");
            root.className = "kk-compare-root";
            root.style.height = `${COMPARE_WIDGET_HEIGHT}px`;

            const toolbar = document.createElement("div");
            toolbar.className = "kk-compare-toolbar";
            const live = document.createElement("span");
            live.className = "kk-compare-live";
            live.textContent = "LIVE";
            const title = document.createElement("span");
            title.className = "kk-compare-title";
            title.textContent = "等待两张输入图像";
            const time = document.createElement("span");
            time.className = "kk-compare-time";
            time.textContent = "--:--:--";
            toolbar.append(live, title, time);

            const stage = document.createElement("div");
            stage.className = "kk-compare-stage";
            const canvas = document.createElement("canvas");
            canvas.className = "kk-compare-canvas";
            const status = document.createElement("div");
            status.className = "kk-compare-status";
            status.textContent = "执行后显示参考图1与调整图2";
            const refLabel = document.createElement("span");
            refLabel.className = "kk-compare-label ref";
            refLabel.textContent = "参考图1";
            const targetLabel = document.createElement("span");
            targetLabel.className = "kk-compare-label target";
            targetLabel.textContent = "调整图2";
            stage.append(canvas, status, refLabel, targetLabel);

            const metrics = document.createElement("div");
            metrics.className = "kk-compare-metrics";
            const table = document.createElement("div");
            table.className = "kk-compare-table";
            const suggestion = document.createElement("div");
            suggestion.className = "kk-compare-suggestion";
            suggestion.textContent = "执行后显示建议调整值";
            const autoGradeTitle = document.createElement("div");
            autoGradeTitle.className = "kk-auto-grade-title";
            autoGradeTitle.textContent = "自动调色参数";
            const autoGradeScore = document.createElement("span");
            autoGradeScore.textContent = "等待求解";
            autoGradeTitle.appendChild(autoGradeScore);
            const autoGradeGrid = document.createElement("div");
            autoGradeGrid.className = "kk-auto-grade-grid";
            metrics.append(table, autoGradeTitle, autoGradeGrid, suggestion);

            const batchBar = document.createElement("div");
            batchBar.className = "kk-compare-batch";
            const previous = document.createElement("button");
            previous.className = "kk-compare-button";
            previous.textContent = "◀";
            const counter = document.createElement("span");
            counter.textContent = "1 / 1";
            const next = document.createElement("button");
            next.className = "kk-compare-button";
            next.textContent = "▶";
            batchBar.append(previous, counter, next);
            root.append(toolbar, stage, metrics, batchBar);

            const widget = node.addDOMWidget("kk_dual_scope", "KKDualScopeCanvas", root, {
                serialize: false,
                hideOnZoom: false,
            });

            const state = { entries: [], index: 0, token: 0 };

            const current = () => state.entries[state.index];

            const draw = () => {
                const entry = current();
                const rect = canvas.getBoundingClientRect();
                const dpr = Math.min(window.devicePixelRatio || 1, 2);
                const width = Math.max(1, Math.floor(rect.width * dpr));
                const height = Math.max(1, Math.floor(rect.height * dpr));
                if (canvas.width !== width || canvas.height !== height) {
                    canvas.width = width;
                    canvas.height = height;
                }
                const ctx = canvas.getContext("2d");
                ctx.fillStyle = "#030405";
                ctx.fillRect(0, 0, width, height);
                if (!entry) return;
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = "high";
                const view = entry.data.comparison_view || "";
                if (view.startsWith("左右")) {
                    refLabel.textContent = "参考图1";
                    targetLabel.textContent = "调整图2";
                    const gap = Math.max(2, Math.round(width * 0.006));
                    const half = (width - gap) / 2;
                    drawContained(ctx, entry.referenceImage, 0, 0, half, height);
                    drawContained(ctx, entry.targetImage, half + gap, 0, half, height);
                    ctx.fillStyle = "#3d454c";
                    ctx.fillRect(half, 0, gap, height);
                } else if (view.startsWith("差异")) {
                    refLabel.textContent = "青色：参考图1独有";
                    targetLabel.textContent = "橙色：调整图2独有";
                    drawContained(ctx, entry.differenceImage, 0, 0, width, height);
                } else {
                    refLabel.textContent = "参考图1";
                    targetLabel.textContent = "调整图2";
                    drawContained(ctx, entry.referenceImage, 0, 0, width, height, 0.82);
                    ctx.globalCompositeOperation = "screen";
                    drawContained(ctx, entry.targetImage, 0, 0, width, height, 0.82);
                    ctx.globalCompositeOperation = "source-over";
                }
            };

            const addCell = (text, className) => {
                const cell = document.createElement("span");
                cell.className = `kk-compare-cell ${className || ""}`;
                cell.textContent = text;
                table.appendChild(cell);
            };

            const updateMetrics = () => {
                table.replaceChildren();
                const entry = current();
                if (!entry) return;
                const data = entry.data.metrics;
                addCell("指标", "name header");
                addCell("参考图1", "ref header");
                addCell("调整图2", "target header");
                addCell("差值", "header");
                const rows = [
                    ["平均亮度", "luma"], ["对比度", "contrast"], ["饱和度", "saturation"],
                    ["红色均值", "red"], ["绿色均值", "green"], ["蓝色均值", "blue"],
                ];
                for (const [label, key] of rows) {
                    const delta = data.delta[key];
                    addCell(label, "name");
                    addCell(data.reference[key].toFixed(2), "ref");
                    addCell(data.target[key].toFixed(2), "target");
                    addCell(`${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`, `delta ${delta > 0 ? "positive" : delta < 0 ? "negative" : ""}`);
                }
                const suggested = data.suggestions;
                const adjustment = data.adjustment;
                autoGradeGrid.replaceChildren();
                const parameterRows = [
                    ["曝光", "exposure", " EV"], ["对比度", "contrast", ""],
                    ["高光", "highlights", ""], ["阴影", "shadows", ""],
                    ["白色", "whites", ""], ["黑色", "blacks", ""],
                    ["色温", "temperature", ""], ["色调", "tint", ""],
                    ["自然饱和度", "vibrance", ""], ["饱和度", "saturation", ""],
                ];
                const parameters = adjustment?.parameters || {};
                for (const [label, key, suffix] of parameterRows) {
                    const item = document.createElement("span");
                    item.className = "kk-auto-grade-item";
                    const name = document.createElement("span");
                    name.textContent = label;
                    const value = document.createElement("b");
                    const numeric = Number(parameters[key] || 0);
                    value.textContent = `${numeric >= 0 ? "+" : ""}${numeric.toFixed(key === "exposure" ? 2 : 1)}${suffix}`;
                    item.append(name, value);
                    autoGradeGrid.appendChild(item);
                }
                if (adjustment) {
                    autoGradeScore.textContent = `匹配改善 ${adjustment.match_score.toFixed(1)}% · ${adjustment.before_error.toFixed(4)} → ${adjustment.after_error.toFixed(4)}`;
                } else {
                    autoGradeScore.textContent = "旧结果，请重新执行";
                }
                const changeText = adjustment
                    ? `图2像素变化：平均 ${adjustment.mean_pixel_change.toFixed(3)}% / 最大 ${adjustment.max_pixel_change.toFixed(3)}%　`
                    : "";
                suggestion.textContent = `${changeText}当前差异建议：曝光 ${suggested.exposure_ev >= 0 ? "+" : ""}${suggested.exposure_ev.toFixed(2)} EV　对比度 ×${suggested.contrast.toFixed(2)}　饱和度 ×${suggested.saturation.toFixed(2)}`;
            };

            const update = () => {
                const count = state.entries.length;
                state.index = Math.min(state.index, Math.max(0, count - 1));
                batchBar.classList.toggle("visible", count > 1);
                counter.textContent = `${state.index + 1} / ${Math.max(1, count)}`;
                previous.disabled = state.index <= 0;
                next.disabled = state.index >= count - 1;
                draw();
                updateMetrics();
            };

            previous.onclick = () => { if (state.index > 0) state.index -= 1; update(); };
            next.onclick = () => { if (state.index < state.entries.length - 1) state.index += 1; update(); };

            const loadPairs = async (pairs) => {
                const token = ++state.token;
                status.style.display = "flex";
                status.textContent = "正在比较两张图像…";
                const entries = await Promise.all((Array.isArray(pairs) ? pairs : []).map(async (pair) => {
                    const load = (info) => new Promise((resolve) => {
                        const image = new Image();
                        image.onload = () => resolve(image);
                        image.onerror = () => resolve(null);
                        image.src = compareImageUrl(info);
                    });
                    const [referenceImage, targetImage, differenceImage] = await Promise.all([
                        load(pair.reference), load(pair.target), load(pair.difference || pair.target),
                    ]);
                    return { data: pair, referenceImage, targetImage, differenceImage };
                }));
                if (token !== state.token) return;
                state.entries = entries;
                if (!entries.length || !entries.some((entry) => entry.referenceImage && entry.targetImage)) {
                    live.classList.remove("active");
                    status.textContent = "对比示波器加载失败";
                    return;
                }
                const first = entries[0].data;
                title.textContent = `${first.scope_type} · ${first.comparison_view}`;
                time.textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
                live.classList.add("active");
                status.style.display = "none";
                update();
                node.setDirtyCanvas(true, true);
            };

            node._kkLoadCompareScopes = loadPairs;
            // Restore outputs when an older serialized node is loaded.
            const expectedOutputs = [
                ["image2_adjusted", "IMAGE"],
                ["grade_parameters", "STRING"],
                ["match_score", "FLOAT"],
            ];
            for (const [name, type] of expectedOutputs) {
                const existing = node.outputs?.find((output) => output.type === type);
                if (existing) {
                    existing.name = name;
                    existing.label = name;
                } else {
                    node.addOutput(name, type);
                }
            }
            widget.computeSize = (width) => [width, COMPARE_WIDGET_HEIGHT];
            requestAnimationFrame(() => {
                migrateLegacyCompareWidgets(node);
                const computed = node.computeSize();
                node.setSize([760, computed[1]]);
                node.setDirtyCanvas(true, true);
            });

            const observer = new ResizeObserver(draw);
            observer.observe(stage);
            chainCompareCallback(node, "onResize", draw);
            chainCompareCallback(node, "onConfigure", function () {
                migrateLegacyCompareWidgets(node);
                // Some frontend versions restore widgets_values immediately
                // after onConfigure. Recheck once on the next animation frame.
                requestAnimationFrame(() => migrateLegacyCompareWidgets(node));
                const saved = node.properties?.kk_compare_scope;
                if (saved) loadPairs(saved);
            });
            chainCompareCallback(node, "onRemoved", function () {
                observer.disconnect();
                state.token += 1;
            });
        });

        chainCompareCallback(nodeType.prototype, "onExecuted", function (message) {
            const pairs = message?.kk_compare_scope;
            if (!pairs?.length) return;
            this.properties = this.properties || {};
            this.properties.kk_compare_scope = pairs;
            this._kkLoadCompareScopes?.(pairs);
        });
    },
});
