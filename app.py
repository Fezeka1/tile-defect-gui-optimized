"""
Streamlit GUI for tile surface defect inspection.

Run with:
    streamlit run app.py

Looks for (in order of preference):
    model_scripted.pt   -- TorchScript-optimized model (faster, see optimize_speed.py)
    optimized_autoencoder.pth -- retrained/optimized checkpoint (see train.py)
    best_baseline_autoencoder.pth -- original baseline checkpoint

And for calibration.json (produced by evaluate.py) to pre-fill the
recommended anomaly threshold. Everything still works without these --
sensible defaults are used and clearly labeled as such.

Also looks for icon.png in this folder for the browser tab icon.
"""
import json
import os
import time

import numpy as np
import streamlit as st
import torch
from PIL import Image

from dataset import encoder_transform, target_transform
from model import load_autoencoder

ICON_PATH = "icon.png"
page_icon = "🔍"
if os.path.exists(ICON_PATH):
    try:
        page_icon = Image.open(ICON_PATH)
    except Exception:
        pass

st.set_page_config(page_title="Tile Defect Inspector", page_icon=page_icon, layout="wide")

CANDIDATE_MODELS = [
    ("model_scripted.pt", "torchscript"),
    ("optimized_autoencoder.pth", "state_dict"),
    ("best_baseline_autoencoder.pth", "state_dict"),
]
CALIBRATION_PATH = "calibration.json"


@st.cache_resource
def load_model():
    device = torch.device("cpu")
    for path, kind in CANDIDATE_MODELS:
        if os.path.exists(path):
            if kind == "torchscript":
                model = torch.jit.load(path, map_location=device)
                model.eval()
            else:
                model = load_autoencoder(path, device=device)
            return model, path
    return None, None


@st.cache_data
def load_calibration():
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    return None


def run_inference(_model, pil_image):
    encoder_input = encoder_transform(pil_image).unsqueeze(0)
    target = target_transform(pil_image).unsqueeze(0)

    start = time.time()
    with torch.no_grad():
        reconstruction = _model(encoder_input)
    latency_ms = (time.time() - start) * 1000

    pixel_error = ((reconstruction - target) ** 2).mean(dim=1).squeeze(0).numpy()  # [H, W]
    image_error = float(pixel_error.mean())

    recon_np = reconstruction.squeeze(0).permute(1, 2, 0).numpy()
    recon_np = np.clip(recon_np, 0, 1)

    return recon_np, pixel_error, image_error, latency_ms


def heatmap_overlay(target_pil, pixel_error):
    import matplotlib

    target_np = np.array(target_pil.resize((128, 128))) / 255.0
    normed = (pixel_error - pixel_error.min()) / (pixel_error.max() - pixel_error.min() + 1e-8)
    heat = matplotlib.colormaps["inferno"](normed)[..., :3]
    overlay = 0.55 * target_np + 0.45 * heat
    return np.clip(overlay, 0, 1)


def collect_uploaded_images(uploaded_files):
    items = []
    for f in uploaded_files:
        pil_image = Image.open(f).convert("RGB")
        identity = f"upload:{f.name}:{f.size}"
        items.append({"name": f.name, "image": pil_image, "identity": identity})
    return items


def render_result(name, pil_image, recon_np, pixel_error, image_error, latency_ms, threshold):
    is_defective = image_error > threshold

    st.subheader(name)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(pil_image, caption="Original", use_container_width=True)
    with col2:
        overlay = heatmap_overlay(pil_image, pixel_error)
        st.image(overlay, caption="Anomaly heatmap", use_container_width=True)
    with col3:
        verdict = "DEFECTIVE" if is_defective else "GOOD"
        color = "red" if is_defective else "green"
        st.markdown(f"### :{color}[{verdict}]")
        margin = abs(image_error - threshold) / threshold * 100
        st.caption(f"{margin:.0f}% {'above' if is_defective else 'below'} the flagging threshold")
        with st.expander("Technical details"):
            st.image(recon_np, caption="Model's expected reconstruction", width=200)
            st.metric("Reconstruction error", f"{image_error:.5f}")
            st.metric("Threshold", f"{threshold:.5f}")
            st.metric("Inference latency", f"{latency_ms:.1f} ms")


def main():
    st.title("🔍 Tile Defect Inspector")
    st.caption("Upload a tile image to check for cracks, chips, and blemishes.")

    model, model_path = load_model()
    if model is None:
        st.error(
            "No model checkpoint found. Place one of "
            f"{[c[0] for c in CANDIDATE_MODELS]} in this directory."
        )
        return

    calibration = load_calibration()
    roc_auc_note = None

    # Youden's J gives the best balance of catching defects vs. false alarms
    # (see calibration.json comparison) -- it's the sane default for an
    # end user, unlike the overly conservative percentile threshold.
    default_threshold = 0.01
    sensitivity_presets = {}
    if calibration:
        roc_auc_note = calibration.get("image_level_roc_auc")
        t = calibration["thresholds"]
        sensitivity_presets = {
            "Lenient (flags fewer tiles, may miss subtle defects)": t["supervised_f1_optimal"],
            "Balanced (recommended)": t["supervised_youden_j"],
            "Strict (flags more tiles, more false alarms)": t["unsupervised_percentile"],
        }
        default_threshold = t["supervised_youden_j"]

    with st.sidebar:
        st.subheader("Sensitivity")
        if sensitivity_presets:
            choice = st.radio(
                "How strict should defect flagging be?",
                options=list(sensitivity_presets.keys()),
                index=1,  # Balanced
                label_visibility="collapsed",
            )
            threshold = sensitivity_presets[choice]
        else:
            threshold = default_threshold

        with st.expander("Advanced (exact threshold)"):
            threshold = st.slider(
                "Reconstruction-error threshold",
                min_value=0.0, max_value=0.02,
                value=float(threshold), step=0.0005, format="%.4f",
            )
            st.caption("Lower = more sensitive. Only adjust this if you know what you're doing "
                       "-- the presets above are calibrated from real defect examples.")

        with st.expander("About this tool"):
            st.write(f"Model file: `{model_path}`")
            if roc_auc_note is not None:
                st.write(f"Validation accuracy score (ROC-AUC): {roc_auc_note:.2f} (1.0 = perfect)")
            else:
                st.caption("No calibration data found — using an uncalibrated default sensitivity.")
            st.caption("Trained only on defect-free tile images. Defects are flagged when the "
                       "model's reconstruction of an image differs enough from the original.")

    # -- Session state: current result(s) shown up front, everything else in History --
    if "history" not in st.session_state:
        st.session_state.history = []       # compact past entries, most recent first
    if "current" not in st.session_state:
        st.session_state.current = None     # {"items": [...], "run_id": ...}

    uploaded_files = st.file_uploader(
        "Upload one or more tile images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    new_items = collect_uploaded_images(uploaded_files) if uploaded_files else []

    if new_items:
        run_id = tuple(item["identity"] for item in new_items)
        if run_id != (st.session_state.current or {}).get("run_id"):
            # A genuinely new upload/scan -- move the previous "current" into
            # history, then compute fresh results for this batch.
            if st.session_state.current:
                for old_item in st.session_state.current["items"]:
                    st.session_state.history.insert(0, {
                        "name": old_item["name"],
                        "thumb": old_item["image"].copy().resize((64, 64)),
                        "image_error": old_item["image_error"],
                    })
                st.session_state.history = st.session_state.history[:200]

            processed = []
            for item in new_items:
                recon_np, pixel_error, image_error, latency_ms = run_inference(model, item["image"])
                processed.append({**item, "recon_np": recon_np, "pixel_error": pixel_error,
                                   "image_error": image_error, "latency_ms": latency_ms})
            st.session_state.current = {"items": processed, "run_id": run_id}

    if not st.session_state.current:
        st.info("Upload an image to run inspection.")
    else:
        st.divider()
        for item in st.session_state.current["items"]:
            render_result(item["name"], item["image"], item["recon_np"], item["pixel_error"],
                          item["image_error"], item["latency_ms"], threshold)
            st.divider()

    if st.session_state.history:
        with st.expander(f"History ({len(st.session_state.history)} previous images)"):
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()
            for entry in st.session_state.history:
                is_defective = entry["image_error"] > threshold
                hcol1, hcol2, hcol3 = st.columns([1, 3, 2])
                with hcol1:
                    st.image(entry["thumb"], width=64)
                with hcol2:
                    st.write(entry["name"])
                with hcol3:
                    color = "red" if is_defective else "green"
                    st.markdown(f":{color}[{'DEFECTIVE' if is_defective else 'GOOD'}]")


if __name__ == "__main__":
    main()