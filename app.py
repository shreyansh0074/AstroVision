import os
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"

import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
from collections import Counter

# 🔥 IMPORTANT: import ultralytics normally
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (NO HACKS)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "clean_model.pt"

    if not model_path.exists():
        st.error("Model file 'clean_model.pt' not found.")
        st.stop()

    return YOLO(str(model_path), task="detect")


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

            st.markdown(f"**Class {cid}** — {count} detected")
            st.caption(
                f"Avg confidence: {sum(confs)/len(confs):.2f} | "
                f"Max confidence: {max(confs):.2f}"
            )
    else:
        st.info("No objects detected.")