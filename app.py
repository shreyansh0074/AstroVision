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
# CBAM CLASSES (same as training file)
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
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


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
# ✅ CRITICAL FIX: Register CBAM module for torch.load
# ══════════════════════════════════════════════════════════════════════════════

cbam_module = types.ModuleType("cbam")
cbam_module.CBAM = CBAM
cbam_module.ChannelAttention = ChannelAttention
cbam_module.SpatialAttention = SpatialAttention

# Only safe mappings (DO NOT override ultralytics modules)
sys.modules["cbam"] = cbam_module
sys.modules["models.common"] = cbam_module
sys.modules["__main__"] = cbam_module


# ══════════════════════════════════════════════════════════════════════════════
# IMPORT ULTRALYTICS
# ══════════════════════════════════════════════════════════════════════════════

from ultralytics import YOLO

# Inject CBAM into ultralytics safely
import ultralytics.nn.modules as _ult_modules
import ultralytics.nn.modules.block as _ult_block
import ultralytics.nn.tasks as _ult_tasks

for _reg in (_ult_modules, _ult_block, _ult_tasks):
    setattr(_reg, "CBAM", CBAM)
    setattr(_reg, "ChannelAttention", ChannelAttention)
    setattr(_reg, "SpatialAttention", SpatialAttention)


# ══════════════════════════════════════════════════════════════════════════════
# CLASS INFO
# ══════════════════════════════════════════════════════════════════════════════

CLASS_INFO = {
    0: {"name": "Comet", "description": "A small icy body with a glowing tail."},
    1: {"name": "Galaxy", "description": "A massive system of stars and dark matter."},
    2: {"name": "Globular Cluster", "description": "A dense cluster of old stars."},
    3: {"name": "Nebula", "description": "A cloud where stars are formed."},
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (clean, no hacks)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best_fixed.pt"

    if not model_path.exists():
        st.error("Model file 'best_fixed.pt' not found.")
        st.stop()

    model = YOLO(str(model_path), task="detect")
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

    st.divider()
    st.subheader("Detected Objects")

    detected_data = [
        (int(box.cls[0]), float(box.conf[0]))
        for box in results[0].boxes
    ]

    if detected_data:
        counts = Counter(cls_id for cls_id, _ in detected_data)

        for cls_id, count in counts.items():
            info = CLASS_INFO.get(cls_id, {
                "name": model.names.get(cls_id, f"Class {cls_id}"),
                "description": ""
            })

            confs = [c for cid, c in detected_data if cid == cls_id]

            st.markdown(f"**{info['name']}** — {count} detected")
            st.caption(
                f"Avg confidence: {sum(confs)/len(confs):.2f} | "
                f"Max confidence: {max(confs):.2f}"
            )

            if info["description"]:
                st.caption(info["description"])

        st.divider()
        st.caption(f"Total objects detected: {len(detected_data)}")
    else:
        st.info("No objects detected.")