import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
from collections import Counter
from pathlib import Path

# ── Class label mapping ────────────────────────────────────────────────────────
# Matches cosmica.yaml: 0=comet, 1=galaxy, 2=globular_cluster, 3=nebula
CLASS_INFO = {
    0: {
        "name": "Comet",
        "description": "A small icy body that releases gas or dust when near the sun, forming a visible tail."
    },
    1: {
        "name": "Galaxy",
        "description": "A massive system of stars, gas, and dark matter bound together by gravity."
    },
    2: {
        "name": "Globular Cluster",
        "description": "A tightly bound sphere of old stars orbiting a galaxy's core."
    },
    3: {
        "name": "Nebula",
        "description": "A cloud of gas and dust in space where stars may form."
    },
}

# ── Model loading ──────────────────────────────────────────────────────────────
# @st.cache_resource ensures the model loads only ONCE across all reruns.
# Without this, Streamlit reloads the model on every interaction — very slow.
# Path is relative to app.py so it works locally and on Streamlit Cloud.
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best_fixed.pt"
    if not model_path.exists():
        st.error(
            f"Model file not found: `{model_path}`\n\n"
            "Make sure `best_fixed.pt` is in the same folder as `app.py`."
        )
        st.stop()
    return YOLO(str(model_path))

model = load_model()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AstroVision", layout="centered")

st.title("🔭 AstroVision")
st.write("Upload an astronomical image to detect objects.")

st.divider()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Input Image", use_container_width=True)

    # ── Prediction ─────────────────────────────────────────────────────────────
    with st.spinner("Detecting objects..."):
        results = model.predict(img_array, conf=0.25, device="cpu")

    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    st.image(annotated_img, caption="Detection Output", use_container_width=True)

    # ── Results summary ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Detected Objects")

    detected_data = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detected_data.append((cls_id, conf))

    if detected_data:
        counts = Counter([cls_id for cls_id, _ in detected_data])

        for cls_id, count in counts.items():
            info = CLASS_INFO.get(
                cls_id,
                {
                    "name": model.names.get(cls_id, f"Class {cls_id}"),
                    "description": ""
                }
            )

            confidences = [conf for cid, conf in detected_data if cid == cls_id]
            avg_conf = sum(confidences) / len(confidences)
            max_conf = max(confidences)

            st.markdown(f"**{info['name']}** — {count} detected")
            st.caption(
                f"Avg confidence: {avg_conf:.2f} | Max confidence: {max_conf:.2f}"
            )
            if info["description"]:
                st.caption(info["description"])

        st.divider()
        st.caption(f"Total objects detected: {len(detected_data)}")
    else:
        st.info(
            "No objects detected. Try a clearer image or a lower confidence threshold."
        )
