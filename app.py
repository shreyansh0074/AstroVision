import os
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"

import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
import sys
import types


# ══════════════════════════════════════════════════════════════════════════════
# CBAM CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(
            self.mlp(self.avg_pool(x)) +
            self.mlp(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# SAFE CBAM REGISTRATION (for torch.load)
# ══════════════════════════════════════════════════════════════════════════════

cbam_module = types.ModuleType("cbam")
cbam_module.CBAM = CBAM
cbam_module.ChannelAttention = ChannelAttention
cbam_module.SpatialAttention = SpatialAttention

sys.modules["cbam"] = cbam_module
sys.modules["models.common"] = cbam_module
sys.modules["__main__"] = cbam_module


# ══════════════════════════════════════════════════════════════════════════════
# IMPORT ULTRALYTICS
# ══════════════════════════════════════════════════════════════════════════════

from ultralytics import YOLO

# Inject CBAM safely (no overriding)
import ultralytics.nn.modules as _ult_modules
import ultralytics.nn.modules.block as _ult_block
import ultralytics.nn.tasks as _ult_tasks

for _mod in (_ult_modules, _ult_block, _ult_tasks):
    setattr(_mod, "CBAM", CBAM)
    setattr(_mod, "ChannelAttention", ChannelAttention)
    setattr(_mod, "SpatialAttention", SpatialAttention)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (SAFE + FALLBACK)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best_fixed.pt"

    if not model_path.exists():
        st.error("Model file not found.")
        st.stop()

    original_load = torch.load

    def safe_load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        try:
            return original_load(*args, **kwargs)
        except ModuleNotFoundError:
            # fallback injection if model was saved with weird module path
            sys.modules["cbam"] = sys.modules[__name__]
            sys.modules["models.common"] = sys.modules[__name__]
            sys.modules["__main__"] = sys.modules[__name__]
            return original_load(*args, **kwargs)

    torch.load = safe_load

    try:
        model = YOLO(str(model_path), task="detect")
    finally:
        torch.load = original_load

    return model


model = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="AstroVision", layout="centered")

st.title("🔭 AstroVision")
st.write("Upload an astronomical image to detect objects.")
st.divider()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Detecting objects..."):
        results = model.predict(img_array, conf=0.25, device="cpu")

    annotated = results[0].plot()[..., ::-1]
    st.image(annotated, caption="Detection Output", use_container_width=True)

    st.subheader("Detected Objects")

    detected = [(int(b.cls[0]), float(b.conf[0])) for b in results[0].boxes]

    if detected:
        counts = Counter(cid for cid, _ in detected)

        for cid, count in counts.items():
            confs = [c for i, c in detected if i == cid]
            st.write(f"Class {cid}: {count} detected")
            st.caption(f"Avg: {sum(confs)/len(confs):.2f} | Max: {max(confs):.2f}")
    else:
        st.info("No objects detected.")