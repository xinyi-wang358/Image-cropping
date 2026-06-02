const DEFAULT_PREVIEW_BOX = {
  left: 91.1154,
  top: 9.73358,
  width: 165.231,
  height: 392.45,
};

const panelsEl = document.getElementById("panels");
const statusEl = document.getElementById("status");
const jsonBox = document.getElementById("jsonBox");
const generateBtn = document.getElementById("generateBtn");
const copyBtn = document.getElementById("copyBtn");
const stitchedEl = document.getElementById("stitched");

const state = Array.from({ length: 3 }, (_, index) => ({
  index,
  file: null,
  detection: null,
  previewImg: null,
  overlay: null,
  bboxRect: null,
  handles: null,
  dragEdge: null,
  croppedDataUrl: null,
}));

function buildDefaultBBox(previewImage) {
  const rect = previewImage.getBoundingClientRect();
  const imageWidth = previewImage.naturalWidth;
  const imageHeight = previewImage.naturalHeight;
  const scaleX = imageWidth / rect.width;
  const scaleY = imageHeight / rect.height;

  const x1 = Math.round(DEFAULT_PREVIEW_BOX.left * scaleX);
  const y1 = Math.round(DEFAULT_PREVIEW_BOX.top * scaleY);
  const x2 = Math.round((DEFAULT_PREVIEW_BOX.left + DEFAULT_PREVIEW_BOX.width) * scaleX);
  const y2 = Math.round((DEFAULT_PREVIEW_BOX.top + DEFAULT_PREVIEW_BOX.height) * scaleY);

  return {
    x1: clamp(x1, 0, Math.max(0, imageWidth - 1)),
    y1: clamp(y1, 0, Math.max(0, imageHeight - 1)),
    x2: clamp(x2, 1, imageWidth),
    y2: clamp(y2, 1, imageHeight),
  };
}

function fixedDetection(previewImage) {
  const defaultBox = buildDefaultBBox(previewImage);
  return {
    source: "manual_fixed",
    fallback_reason: "auto_pipeline_disabled",
    bbox_detector: { ...defaultBox },
    bbox_qwen: null,
    bbox_final: { ...defaultBox },
    waist_y: defaultBox.y1,
    hem_y: defaultBox.y2,
    confidence: 0.99,
    detector_debug: {
      mode: "fixed_preview_bbox",
      note: "Default bbox is converted from the fixed preview box.",
    },
  };
}

function updateJson() {
  jsonBox.textContent = JSON.stringify(state.map((item) => item.detection), null, 2);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function setStatus(text) {
  statusEl.textContent = text;
}

function requireStitchedImage() {
  if (!stitchedEl.src) {
    throw new Error("请先生成拼接图。");
  }
  return stitchedEl.src;
}

async function imageUrlToPngBlob(imageUrl) {
  const image = new Image();
  image.decoding = "async";
  image.src = imageUrl;

  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error("图片加载失败，无法复制。"));
  });

  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("当前浏览器无法创建图片画布。");
  }
  context.drawImage(image, 0, 0);

  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("当前浏览器无法生成 PNG 图片。"));
      }
    }, "image/png");
  });
}

function createPanel(index) {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `
    <h3>图片 ${index + 1}</h3>
    <input id="file${index}" type="file" accept="image/*">
    <div class="panelNote">支持拖拽上传。上传后会立即显示固定红框。</div>
    <div class="previewWrap">
      <img class="preview" id="img${index}" alt="图片 ${index + 1} 预览">
      <div class="overlay" id="ov${index}">
        <div class="bbox" id="box${index}"></div>
        <div class="edge left" data-i="${index}" data-edge="left"></div>
        <div class="edge right" data-i="${index}" data-edge="right"></div>
        <div class="edge top" data-i="${index}" data-edge="top"></div>
        <div class="edge bottom" data-i="${index}" data-edge="bottom"></div>
      </div>
    </div>
  `;
  panelsEl.appendChild(panel);

  const fileInput = panel.querySelector(`#file${index}`);
  const bindFile = (file) => {
    state[index].file = file || null;
    state[index].croppedDataUrl = null;
    stitchedEl.removeAttribute("src");

    if (!file) {
      state[index].detection = null;
      state[index].previewImg.removeAttribute("src");
      state[index].overlay.style.display = "none";
      updateJson();
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    state[index].previewImg.onload = () => {
      state[index].detection = fixedDetection(state[index].previewImg);
      renderOverlay(index);
      updateJson();
    };
    state[index].previewImg.src = objectUrl;
  };

  fileInput.addEventListener("change", () => bindFile(fileInput.files[0]));
  panel.addEventListener("dragover", (event) => {
    event.preventDefault();
    panel.classList.add("dragover");
  });
  panel.addEventListener("dragleave", () => panel.classList.remove("dragover"));
  panel.addEventListener("drop", (event) => {
    event.preventDefault();
    panel.classList.remove("dragover");
    const file = event.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) {
      bindFile(file);
    }
  });

  state[index].previewImg = panel.querySelector(`#img${index}`);
  state[index].overlay = panel.querySelector(`#ov${index}`);
  state[index].bboxRect = panel.querySelector(`#box${index}`);
  state[index].handles = Array.from(panel.querySelectorAll(".edge"));
}

function renderOverlay(index) {
  const item = state[index];
  if (!item.detection || !item.previewImg.naturalWidth) return;

  const rect = item.previewImg.getBoundingClientRect();
  const scaleX = rect.width / item.previewImg.naturalWidth;
  const scaleY = rect.height / item.previewImg.naturalHeight;
  const box = item.detection.bbox_final;
  const x1 = box.x1 * scaleX;
  const y1 = box.y1 * scaleY;
  const x2 = box.x2 * scaleX;
  const y2 = box.y2 * scaleY;

  item.overlay.style.display = "block";
  item.bboxRect.style.left = `${x1}px`;
  item.bboxRect.style.top = `${y1}px`;
  item.bboxRect.style.width = `${x2 - x1}px`;
  item.bboxRect.style.height = `${y2 - y1}px`;

  item.handles.forEach((handle) => {
    const edge = handle.dataset.edge;
    if (edge === "left") {
      handle.style.left = `${x1 - 5}px`;
      handle.style.top = `${y1}px`;
      handle.style.height = `${y2 - y1}px`;
      handle.style.width = "10px";
    }
    if (edge === "right") {
      handle.style.left = `${x2 - 5}px`;
      handle.style.top = `${y1}px`;
      handle.style.height = `${y2 - y1}px`;
      handle.style.width = "10px";
    }
    if (edge === "top") {
      handle.style.top = `${y1 - 5}px`;
      handle.style.left = `${x1}px`;
      handle.style.width = `${x2 - x1}px`;
      handle.style.height = "10px";
    }
    if (edge === "bottom") {
      handle.style.top = `${y2 - 5}px`;
      handle.style.left = `${x1}px`;
      handle.style.width = `${x2 - x1}px`;
      handle.style.height = "10px";
    }
  });
}

function startDrag(event) {
  const target = event.target;
  if (!target.classList.contains("edge")) return;

  const index = Number(target.dataset.i);
  state[index].dragEdge = target.dataset.edge;
  target.setPointerCapture(event.pointerId);
}

function moveDrag(event) {
  for (const item of state) {
    if (!item.dragEdge || !item.detection || !item.previewImg.naturalWidth) continue;

    const rect = item.previewImg.getBoundingClientRect();
    const x = clamp(event.clientX - rect.left, 0, rect.width);
    const y = clamp(event.clientY - rect.top, 0, rect.height);
    const px = (x / rect.width) * item.previewImg.naturalWidth;
    const py = (y / rect.height) * item.previewImg.naturalHeight;
    const box = item.detection.bbox_final;

    if (item.dragEdge === "left") box.x1 = Math.min(Math.round(px), box.x2 - 5);
    if (item.dragEdge === "right") box.x2 = Math.max(Math.round(px), box.x1 + 5);
    if (item.dragEdge === "top") {
      const nextY = Math.min(Math.round(py), box.y2 - 5);
      box.y1 = nextY;
      item.detection.waist_y = nextY;
    }
    if (item.dragEdge === "bottom") {
      const nextY = Math.max(Math.round(py), box.y1 + 5);
      box.y2 = nextY;
      item.detection.hem_y = nextY;
    }

    renderOverlay(item.index);
  }
  updateJson();
}

function stopDrag() {
  state.forEach((item) => {
    item.dragEdge = null;
  });
}

async function generateComposite() {
  try {
    setStatus("裁剪并拼接中...");
    for (const item of state) {
      if (!item.file) throw new Error(`请先上传图片 ${item.index + 1}`);
    }

    for (const item of state) {
      const form = new FormData();
      form.append("image", item.file);
      form.append("bbox", JSON.stringify(item.detection.bbox_final));
      const response = await fetch("/debug/crop", { method: "POST", body: form });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      item.croppedDataUrl = payload.cropped_image_data_url;
    }

    const stitchedResponse = await fetch("/debug/stitch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ images: state.map((item) => item.croppedDataUrl) }),
    });
    if (!stitchedResponse.ok) throw new Error(await stitchedResponse.text());

    const stitchedPayload = await stitchedResponse.json();
    stitchedEl.src = stitchedPayload.stitched_image_data_url;
    updateJson();
    setStatus("已完成：固定框裁剪并拼接。");
  } catch (error) {
    setStatus("处理失败。");
    jsonBox.textContent = String(error);
  }
}

async function copyComposite() {
  try {
    const imageUrl = requireStitchedImage();
    if (!navigator.clipboard || !window.ClipboardItem) {
      throw new Error("当前浏览器不支持图片剪贴板。请用 Chrome 或 Edge 打开本页。");
    }

    const blob = await imageUrlToPngBlob(imageUrl);
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    setStatus("已复制拼接图到剪贴板。");
  } catch (error) {
    setStatus(`复制失败：${error.message || error}`);
    jsonBox.textContent = String(error);
  }
}

for (let index = 0; index < 3; index += 1) createPanel(index);

document.addEventListener("pointerdown", startDrag);
document.addEventListener("pointermove", moveDrag);
document.addEventListener("pointerup", stopDrag);
window.addEventListener("resize", () => {
  state.forEach((item) => {
    if (item.file && item.detection) renderOverlay(item.index);
  });
});
generateBtn.addEventListener("click", generateComposite);
copyBtn.addEventListener("click", copyComposite);

updateJson();
