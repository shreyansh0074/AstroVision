import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
from collections import Counter
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# CBAM — must be defined AND registered BEFORE any YOLO import touches weights
# ══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        hidden = max(1, in_channels // reduction_ratio)
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.max_pool   = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(
            self.shared_mlp(self.avg_pool(x)) +
            self.shared_mlp(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1,  keepdim=True).values
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


# ── Register CBAM in every place ultralytics looks for custom modules ──────────
import ultralytics.nn.modules as _ult_modules
import ultralytics.nn.modules.block as _ult_block
import ultralytics.nn.tasks as _ult_tasks

for _registry in (_ult_modules, _ult_block, _ult_tasks):
    setattr(_registry, "CBAM",             CBAM)
    setattr(_registry, "ChannelAttention", ChannelAttention)
    setattr(_registry, "SpatialAttention", SpatialAttention)

import sys as _sys
_sys.modules[__name__].CBAM             = CBAM
_sys.modules[__name__].ChannelAttention = ChannelAttention
_sys.modules[__name__].SpatialAttention = SpatialAttention


# ══════════════════════════════════════════════════════════════════════════════
# Class label mapping  (0=comet, 1=galaxy, 2=globular_cluster, 3=nebula)
# ══════════════════════════════════════════════════════════════════════════════

CLASS_INFO = {
    0: {
        "name":        "Comet",
        "description": "A small icy body that releases gas or dust when near the sun, forming a visible tail."
    },
    1: {
        "name":        "Galaxy",
        "description": "A massive system of stars, gas, and dark matter bound together by gravity."
    },
    2: {
        "name":        "Globular Cluster",
        "description": "A tightly bound sphere of old stars orbiting a galaxy's core."
    },
    3: {
        "name":        "Nebula",
        "description": "A cloud of gas and dust in space where stars may form."
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Model loading — cached so it only runs once per Streamlit session
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best_fixed.pt"
    if not model_path.exists():
        st.error(
            "**Model file not found:** `best_fixed.pt`\n\n"
            "Make sure it is committed to the repo in the same folder as `app.py`."
        )
        st.stop()
    return YOLO(str(model_path))

model = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="AstroVision", layout="centered")

st.title("🔭 AstroVision")
st.write("Upload an astronomical image to detect objects.")

st.divider()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image     = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Detecting objects..."):
        results = model.predict(img_array, conf=0.25, device="cpu")

    annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    st.image(annotated_img, caption="Detection Output", use_container_width=True)

    st.divider()
    st.subheader("Detected Objects")

    detected_data = [
        (int(box.cls[0]), float(box.conf[0]))
        for box in results[0].boxes
    ]

    if detected_data:
        counts = Counter(cls_id for cls_id, _ in detected_data)

        for cls_id, count in counts.items():
            info        = CLASS_INFO.get(cls_id, {
                "name":        model.names.get(cls_id, f"Class {cls_id}"),
                "description": ""
            })
            confidences = [c for cid, c in detected_data if cid == cls_id]
            avg_conf    = sum(confidences) / len(confidences)
            max_conf    = max(confidences)

            st.markdown(f"**{info['name']}** — {count} detected")
            st.caption(f"Avg confidence: {avg_conf:.2f} | Max confidence: {max_conf:.2f}")
            if info["description"]:
                st.caption(info["description"])

        st.divider()
        st.caption(f"Total objects detected: {len(detected_data)}")

    else:
        st.info("No objects detected. Try a clearer image or a lower confidence threshold.")