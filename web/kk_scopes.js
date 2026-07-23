const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const STYLE_ID = "kk-davinci-scopes-style";
const SCOPE_WIDGET_HEIGHT = 360;

function installStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .kk-scope-root {
            width: 100%; height: ${SCOPE_WIDGET_HEIGHT}px;
            min-height: ${SCOPE_WIDGET_HEIGHT}px;
            max-height: ${SCOPE_WIDGET_HEIGHT}px;
            display: flex; flex-direction: column; overflow: hidden;
            background: #090b0d; border: 1px solid #30343a;
            box-sizing: border-box; user-select: none;
        }
        .kk-scope-toolbar {
            height: 30px; flex: 0 0 30px; display: flex;
            align-items: center; gap: 8px; padding: 0 8px;
            color: #c7ccd2; background: #1b1e22;
            border-bottom: 1px solid #30343a; box-sizing: border-box;
            font: 12px/1.2 sans-serif;
        }
        .kk-scope-live {
            display: inline-flex; align-items: center; gap: 5px;
            color: #778089; font-size: 10px; font-weight: 700;
            letter-spacing: .6px;
        }
        .kk-scope-live::before {
            content: ""; width: 7px; height: 7px; border-radius: 50%;
            background: #555c63; box-shadow: 0 0 0 transparent;
        }
        .kk-scope-live.active { color: #65e891; }
        .kk-scope-live.active::before {
            background: #35dd72; box-shadow: 0 0 8px #35dd72;
        }
        .kk-scope-title {
            flex: 1 1 auto; overflow: hidden; white-space: nowrap;
            text-overflow: ellipsis; text-align: center;
        }
        .kk-scope-time { color: #747d86; font-size: 10px; min-width: 48px; text-align: right; }
        .kk-scope-stage { position: relative; flex: 1 1 auto; min-height: 0; background: #050607; }
        .kk-scope-canvas { display: block; width: 100%; height: 100%; }
        .kk-scope-status {
            position: absolute; inset: 0; display: flex; align-items: center;
            justify-content: center; color: #69727b; pointer-events: none;
            font: 12px sans-serif;
        }
        .kk-scope-batch {
            display: none; height: 28px; flex: 0 0 28px; align-items: center;
            justify-content: center; gap: 10px; background: #15181b;
            border-top: 1px solid #30343a; color: #aab1b8;
            font: 11px sans-serif;
        }
        .kk-scope-batch.visible { display: flex; }
        .kk-scope-button {
            width: 28px; height: 20px; border: 1px solid #3c4248;
            border-radius: 3px; background: #252a2f; color: #d3d7dc;
            cursor: pointer; line-height: 16px;
        }
        .kk-scope-button:hover { background: #343b42; }
        .kk-scope-button:disabled { opacity: .35; cursor: default; }
    `;
    document.head.appendChild(style);
}

function chainCallback(target, name, callback) {
    const original = target[name];
    target[name] = function (...args) {
        const result = original?.apply(this, args);
        callback.apply(this, args);
        return result;
    };
}

function scopeUrl(info) {
    const params = new URLSearchParams({
        filename: info.filename,
        subfolder: info.subfolder || "",
        type: info.type || "temp",
        t: String(Date.now()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

app.registerExtension({
    name: "KK.DynamicDaVinciScopes",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "KKDaVinciScopes") return;

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            installStyles();
            const node = this;

            const root = document.createElement("div");
            root.className = "kk-scope-root";
            root.style.height = `${SCOPE_WIDGET_HEIGHT}px`;

            const toolbar = document.createElement("div");
            toolbar.className = "kk-scope-toolbar";
            const live = document.createElement("span");
            live.className = "kk-scope-live";
            live.textContent = "LIVE";
            const title = document.createElement("span");
            title.className = "kk-scope-title";
            title.textContent = "等待输入图像";
            const time = document.createElement("span");
            time.className = "kk-scope-time";
            time.textContent = "--:--:--";
            toolbar.append(live, title, time);

            const stage = document.createElement("div");
            stage.className = "kk-scope-stage";
            const canvas = document.createElement("canvas");
            canvas.className = "kk-scope-canvas";
            const status = document.createElement("div");
            status.className = "kk-scope-status";
            status.textContent = "执行工作流后开始监看";
            stage.append(canvas, status);

            const batch = document.createElement("div");
            batch.className = "kk-scope-batch";
            const previous = document.createElement("button");
            previous.className = "kk-scope-button";
            previous.textContent = "◀";
            const counter = document.createElement("span");
            counter.textContent = "1 / 1";
            const next = document.createElement("button");
            next.className = "kk-scope-button";
            next.textContent = "▶";
            batch.append(previous, counter, next);
            root.append(toolbar, stage, batch);

            const widget = node.addDOMWidget("kk_dynamic_scope", "KKScopeCanvas", root, {
                serialize: false,
                hideOnZoom: false,
            });

            const state = {
                descriptors: [],
                images: [],
                index: 0,
                drawToken: 0,
            };

            const draw = () => {
                const image = state.images[state.index];
                const rect = canvas.getBoundingClientRect();
                const dpr = Math.min(window.devicePixelRatio || 1, 2);
                const pixelWidth = Math.max(1, Math.floor(rect.width * dpr));
                const pixelHeight = Math.max(1, Math.floor(rect.height * dpr));
                if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
                    canvas.width = pixelWidth;
                    canvas.height = pixelHeight;
                }
                const ctx = canvas.getContext("2d");
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.fillStyle = "#050607";
                ctx.fillRect(0, 0, pixelWidth, pixelHeight);
                if (!image) return;
                const scale = Math.min(pixelWidth / image.naturalWidth, pixelHeight / image.naturalHeight);
                const drawWidth = image.naturalWidth * scale;
                const drawHeight = image.naturalHeight * scale;
                const x = (pixelWidth - drawWidth) / 2;
                const y = (pixelHeight - drawHeight) / 2;
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = "high";
                ctx.drawImage(image, x, y, drawWidth, drawHeight);
            };

            const updateBatch = () => {
                const count = state.descriptors.length;
                batch.classList.toggle("visible", count > 1);
                counter.textContent = `${state.index + 1} / ${Math.max(count, 1)}`;
                previous.disabled = state.index <= 0;
                next.disabled = state.index >= count - 1;
                draw();
            };

            previous.onclick = () => {
                if (state.index > 0) state.index -= 1;
                updateBatch();
            };
            next.onclick = () => {
                if (state.index < state.descriptors.length - 1) state.index += 1;
                updateBatch();
            };

            const loadScopes = async (descriptors) => {
                const token = ++state.drawToken;
                state.descriptors = Array.isArray(descriptors) ? descriptors : [];
                state.index = Math.min(state.index, Math.max(0, state.descriptors.length - 1));
                status.style.display = "flex";
                status.textContent = "正在更新示波器…";

                const images = await Promise.all(state.descriptors.map((info) => new Promise((resolve) => {
                    const image = new Image();
                    image.onload = () => resolve(image);
                    image.onerror = () => resolve(null);
                    image.src = scopeUrl(info);
                })));
                if (token !== state.drawToken) return;
                state.images = images;
                const first = state.descriptors[0];
                if (!first || !images.some(Boolean)) {
                    live.classList.remove("active");
                    status.textContent = "示波器数据加载失败";
                    return;
                }
                title.textContent = first.scope_type || "KK达芬奇示波器";
                time.textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
                live.classList.add("active");
                status.style.display = "none";
                updateBatch();
                node.setDirtyCanvas(true, true);
            };

            node._kkLoadScopes = loadScopes;
            // This must not depend on node.size. LiteGraph computes node.size
            // from widget sizes, so feeding node.size back here creates an
            // unbounded height loop when a node is cloned or configured.
            widget.computeSize = (width) => [width, SCOPE_WIDGET_HEIGHT];

            requestAnimationFrame(() => {
                const computed = node.computeSize();
                node.setSize([640, computed[1]]);
                node.setDirtyCanvas(true, true);
            });

            const resizeObserver = new ResizeObserver(draw);
            resizeObserver.observe(stage);

            chainCallback(node, "onResize", draw);
            chainCallback(node, "onConfigure", function () {
                const saved = node.properties?.kk_scope;
                if (saved) loadScopes(saved);
            });
            chainCallback(node, "onRemoved", function () {
                resizeObserver.disconnect();
                state.drawToken += 1;
            });
        });

        chainCallback(nodeType.prototype, "onExecuted", function (message) {
            const descriptors = message?.kk_scope;
            if (!descriptors?.length) return;
            this.properties = this.properties || {};
            this.properties.kk_scope = descriptors;
            this._kkLoadScopes?.(descriptors);
        });
    },
});
