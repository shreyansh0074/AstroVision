import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter


# ══════════════════════════════════════════════════════════════════════════════
# CBAM classes — defined first, before any ultralytics import
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
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


# ══════════════════════════════════════════════════════════════════════════════
# Patch torch's unpickler BEFORE ultralytics is imported
# This is the only reliable way to make custom classes loadable from .pt files
# ══════════════════════════════════════════════════════════════════════════════

import sys
import importlib

# Create a fake module that torch's pickle can resolve CBAM from
class _FakeModule:
    pass

_fake = _FakeModule()
_fake.CBAM             = CBAM
_fake.ChannelAttention = ChannelAttention
_fake.SpatialAttention = SpatialAttention

# Register under every module name the checkpoint might reference
for _mod_name in [
    "ultralytics.nn.modules",
    "ultralytics.nn.modules.block",
    "ultralytics.nn.tasks",
    "models.common",       # older ultralytics checkpoints
    "__main__",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _fake
    else:
        setattr(sys.modules[_mod_name], "CBAM",             CBAM)
        setattr(sys.modules[_mod_name], "ChannelAttention", ChannelAttention)
        setattr(sys.modules[_mod_name], "SpatialAttention", SpatialAttention)

# Now safe to import ultralytics
from ultralytics import YOLO
import ultralytics.nn.modules as _ult_modules
import ultralytics.nn.modules.block as _ult_block
import ultralytics.nn.tasks as _ult_tasks

for _reg in (_ult_modules, _ult_block, _ult_tasks):
    setattr(_reg, "CBAM",             CBAM)
    setattr(_reg, "ChannelAttention", ChannelAttention)
    setattr(_reg, "SpatialAttention", SpatialAttention)


# ══════════════════════════════════════════════════════════════════════════════
# Class label mapping
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
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best_fixed.pt"
    if not model_path.exists():
        st.error("**Model file not found:** `best_fixed.pt` — make sure it is committed to the repo.")
        st.stop()

    # Patch safe_load to allow our custom classes
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        mdl = YOLO(str(model_path))
    finally:
        torch.load = original_load  # always restore

    return mdl

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

    # BGR → RGB without cv2
    annotated_img = results[0].plot()[..., ::-1]
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