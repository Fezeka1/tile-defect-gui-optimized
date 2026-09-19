"""
Tile Defect Inspector GUI.

Run:

    streamlit run app.py

Final configuration:

    Model:
        optimization2_model.pth

    Image size:
        128 x 128

    Scoring:
        top1_mean

    Balanced threshold:
        0.0525
"""

import json
import os
import time

import numpy as np
import streamlit as st
import torch

from PIL import Image

from dataset import (
    encoder_transform,
    target_transform
)

from model import (
    load_autoencoder
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "optimization2_model.pth"
)


CALIBRATION_PATH = os.path.join(
    BASE_DIR,
    "calibration.json"
)


ICON_PATH = os.path.join(
    BASE_DIR,
    "icon.png"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

page_icon = "🔍"


if os.path.exists(ICON_PATH):

    try:

        page_icon = Image.open(
            ICON_PATH
        )

    except Exception:

        pass


st.set_page_config(
    page_title="Tile Defect Inspector",
    page_icon=page_icon,
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cpu"
    )


    if not os.path.exists(
        MODEL_PATH
    ):

        return None, None


    model = load_autoencoder(
        MODEL_PATH,
        device=device
    )


    return model, MODEL_PATH


# ============================================================
# LOAD CALIBRATION
# ============================================================

@st.cache_data
def load_calibration():

    if os.path.exists(
        CALIBRATION_PATH
    ):

        with open(
            CALIBRATION_PATH,
            "r"
        ) as file:

            return json.load(
                file
            )


    return None


# ============================================================
# IMAGE SCORE
# ============================================================

def calculate_image_score(
    pixel_error,
    scoring_method="top1_mean"
):
    """
    Calculate image anomaly score.

    top1_mean:
        Average of highest-error 1% of pixels.

    mean:
        Average reconstruction error over entire image.
    """


    if scoring_method == "top1_mean":

        flat = pixel_error.reshape(
            -1
        )


        top_count = max(
            1,
            int(
                np.ceil(
                    flat.size
                    *
                    0.01
                )
            )
        )


        highest_errors = np.partition(
            flat,
            -top_count
        )[-top_count:]


        return float(
            highest_errors.mean()
        )


    return float(
        pixel_error.mean()
    )


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    _model,
    pil_image,
    scoring_method
):

    encoder_input = (
        encoder_transform(
            pil_image
        )
        .unsqueeze(0)
    )


    target = (
        target_transform(
            pil_image
        )
        .unsqueeze(0)
    )


    start = time.time()


    with torch.no_grad():

        reconstruction = _model(
            encoder_input
        )


    latency_ms = (
        time.time()
        -
        start
    ) * 1000


    # --------------------------------------------------------
    # Reconstruction error
    # --------------------------------------------------------

    pixel_error = (
        (
            reconstruction
            -
            target
        ) ** 2
    ).mean(
        dim=1
    ).squeeze(
        0
    ).cpu().numpy()


    # --------------------------------------------------------
    # Image anomaly score
    # --------------------------------------------------------

    image_error = calculate_image_score(
        pixel_error,
        scoring_method
    )


    # --------------------------------------------------------
    # Reconstruction visualization
    # --------------------------------------------------------

    recon_np = (
        reconstruction
        .squeeze(0)
        .permute(
            1,
            2,
            0
        )
        .cpu()
        .numpy()
    )


    recon_np = np.clip(
        recon_np,
        0,
        1
    )


    return (
        recon_np,
        pixel_error,
        image_error,
        latency_ms
    )


# ============================================================
# HEATMAP
# ============================================================

def heatmap_overlay(
    target_pil,
    pixel_error
):

    import matplotlib


    target_np = (
        np.array(
            target_pil.resize(
                (128, 128)
            )
        )
        /
        255.0
    )


    normed = (
        pixel_error
        -
        pixel_error.min()
    ) / (
        pixel_error.max()
        -
        pixel_error.min()
        +
        1e-8
    )


    heat = (
        matplotlib
        .colormaps[
            "inferno"
        ](
            normed
        )
        [..., :3]
    )


    overlay = (
        0.55
        *
        target_np

        +

        0.45
        *
        heat
    )


    return np.clip(
        overlay,
        0,
        1
    )


# ============================================================
# UPLOAD PROCESSING
# ============================================================

def collect_uploaded_images(
    uploaded_files
):

    items = []


    for file in uploaded_files:

        pil_image = (
            Image.open(file)
            .convert("RGB")
        )


        identity = (
            f"upload:"
            f"{file.name}:"
            f"{file.size}"
        )


        items.append({
            "name": file.name,
            "image": pil_image,
            "identity": identity
        })


    return items


# ============================================================
# RESULT DISPLAY
# ============================================================

def render_result(
    name,
    pil_image,
    recon_np,
    pixel_error,
    image_error,
    latency_ms,
    threshold
):

    is_defective = (
        image_error > threshold
    )


    st.subheader(
        name
    )


    col1, col2, col3 = st.columns(
        3
    )


    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    with col1:

        st.image(
            pil_image,
            caption="Original",
            use_container_width=True
        )


    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    with col2:

        overlay = heatmap_overlay(
            pil_image,
            pixel_error
        )


        st.image(
            overlay,
            caption="Anomaly heatmap",
            use_container_width=True
        )


    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    with col3:

        verdict = (
            "DEFECTIVE"
            if is_defective
            else
            "GOOD"
        )


        color = (
            "red"
            if is_defective
            else
            "green"
        )


        st.markdown(
            f"### :{color}[{verdict}]"
        )


        if threshold > 0:

            margin = (
                abs(
                    image_error
                    -
                    threshold
                )
                /
                threshold
                *
                100
            )

        else:

            margin = 0


        st.caption(
            f"{margin:.0f}% "
            f"{'above' if is_defective else 'below'} "
            f"the flagging threshold"
        )


        with st.expander(
            "Technical details"
        ):

            st.image(
                recon_np,
                caption=(
                    "Model's expected reconstruction"
                ),
                width=200
            )


            st.metric(
                "Reconstruction error",
                f"{image_error:.5f}"
            )


            st.metric(
                "Threshold",
                f"{threshold:.5f}"
            )


            st.metric(
                "Inference latency",
                f"{latency_ms:.1f} ms"
            )


# ============================================================
# MAIN APP
# ============================================================

def main():

    st.title(
        "🔍 Tile Defect Inspector"
    )


    st.caption(
        "Upload a tile image to check for cracks, "
        "chips, and blemishes."
    )


    # ========================================================
    # MODEL
    # ========================================================

    model, model_path = load_model()


    if model is None:

        st.error(
            "Optimization 2 model not found. "
            "Place optimization2_model.pth "
            "in this directory."
        )

        return


    # ========================================================
    # CALIBRATION
    # ========================================================

    calibration = load_calibration()


    scoring_method = "top1_mean"


    roc_auc_note = None


    # --------------------------------------------------------
    # Optimization 2 thresholds
    # --------------------------------------------------------

    f1_threshold = (
        0.05121959000825882
    )


    # YOUR CHOSEN BALANCED THRESHOLD
    balanced_threshold = 0.0525


    conservative_threshold = (
        0.06580093316733837
    )


    # --------------------------------------------------------
    # Read relevant calibration information
    # --------------------------------------------------------

    if calibration:

        scoring_method = calibration.get(
            "recommended_scoring_method",
            "top1_mean"
        )


        method_data = calibration.get(
            scoring_method,
            {}
        )


        roc_auc_note = method_data.get(
            "roc_auc"
        )


        thresholds = method_data.get(
            "thresholds",
            {}
        )


        if thresholds:

            f1_threshold = thresholds.get(
                "supervised_f1_optimal",
                f1_threshold
            )


            conservative_threshold = thresholds.get(
                "unsupervised_percentile",
                conservative_threshold
            )


    # ========================================================
    # PRESETS
    # ========================================================

    sensitivity_presets = {

        "Lenient (flags fewer tiles, may miss subtle defects)":
            conservative_threshold,


        "Balanced (recommended)":
            balanced_threshold,


        "Strict (flags more tiles, more false alarms)":
            f1_threshold,
    }


    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.subheader(
            "Sensitivity"
        )


        choice = st.radio(
            "How strict should defect flagging be?",
            options=list(
                sensitivity_presets.keys()
            ),
            index=1,
            label_visibility="collapsed"
        )


        threshold = (
            sensitivity_presets[
                choice
            ]
        )


        with st.expander(
            "Advanced (exact threshold)"
        ):

            threshold = st.slider(
                "Reconstruction-error threshold",
                min_value=0.0,
                max_value=0.10,
                value=float(
                    threshold
                ),
                step=0.0005,
                format="%.4f"
            )


            st.caption(
                "Lower = more sensitive. "
                "Only adjust this if you know what "
                "you're doing -- the presets above "
                "are calibrated from real defect examples."
            )


        with st.expander(
            "About this tool"
        ):

            st.write(
                f"Model file: "
                f"`{os.path.basename(model_path)}`"
            )


            if roc_auc_note is not None:

                st.write(
                    "Validation accuracy score "
                    "(ROC-AUC): "
                    f"{roc_auc_note:.2f} "
                    "(1.0 = perfect)"
                )


            else:

                st.caption(
                    "No calibration data found."
                )


            st.caption(
                "Trained only on defect-free tile images. "
                "Defects are flagged when the model's "
                "reconstruction of an image differs enough "
                "from the original."
            )


    # ========================================================
    # SESSION STATE
    # ========================================================

    if "history" not in st.session_state:

        st.session_state.history = []


    if "current" not in st.session_state:

        st.session_state.current = None


    # ========================================================
    # UPLOAD
    # ========================================================

    uploaded_files = st.file_uploader(
        "Upload one or more tile images",
        type=[
            "png",
            "jpg",
            "jpeg"
        ],
        accept_multiple_files=True
    )


    new_items = (
        collect_uploaded_images(
            uploaded_files
        )

        if uploaded_files

        else []
    )


    # ========================================================
    # PROCESS NEW IMAGES
    # ========================================================

    if new_items:

        run_id = tuple(
            item["identity"]

            for item
            in new_items
        )


        previous_run_id = (
            (
                st.session_state.current
                or {}
            )
            .get(
                "run_id"
            )
        )


        if run_id != previous_run_id:


            # ------------------------------------------------
            # Move current result to history
            # ------------------------------------------------

            if st.session_state.current:

                for old_item in (
                    st.session_state
                    .current[
                        "items"
                    ]
                ):

                    st.session_state.history.insert(
                        0,
                        {
                            "name":
                                old_item[
                                    "name"
                                ],

                            "thumb":
                                old_item[
                                    "image"
                                ]
                                .copy()
                                .resize(
                                    (64, 64)
                                ),

                            "image_error":
                                old_item[
                                    "image_error"
                                ]
                        }
                    )


                st.session_state.history = (
                    st.session_state
                    .history[:200]
                )


            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

            processed = []


            for item in new_items:

                (
                    recon_np,
                    pixel_error,
                    image_error,
                    latency_ms

                ) = run_inference(
                    model,
                    item["image"],
                    scoring_method
                )


                processed.append({
                    **item,

                    "recon_np":
                        recon_np,

                    "pixel_error":
                        pixel_error,

                    "image_error":
                        image_error,

                    "latency_ms":
                        latency_ms
                })


            st.session_state.current = {
                "items":
                    processed,

                "run_id":
                    run_id
            }


    # ========================================================
    # CURRENT RESULT
    # ========================================================

    if not st.session_state.current:

        st.info(
            "Upload an image to run inspection."
        )


    else:

        st.divider()


        for item in (
            st.session_state
            .current[
                "items"
            ]
        ):

            render_result(
                item["name"],
                item["image"],
                item["recon_np"],
                item["pixel_error"],
                item["image_error"],
                item["latency_ms"],
                threshold
            )


            st.divider()


    # ========================================================
    # HISTORY
    # ========================================================

    if st.session_state.history:

        with st.expander(
            f"History "
            f"({len(st.session_state.history)} "
            f"previous images)"
        ):

            if st.button(
                "Clear history"
            ):

                st.session_state.history = []

                st.rerun()


            for entry in (
                st.session_state.history
            ):

                is_defective = (
                    entry[
                        "image_error"
                    ]
                    >
                    threshold
                )


                hcol1, hcol2, hcol3 = (
                    st.columns(
                        [1, 3, 2]
                    )
                )


                with hcol1:

                    st.image(
                        entry[
                            "thumb"
                        ],
                        width=64
                    )


                with hcol2:

                    st.write(
                        entry[
                            "name"
                        ]
                    )


                with hcol3:

                    color = (
                        "red"
                        if is_defective
                        else
                        "green"
                    )


                    verdict = (
                        "DEFECTIVE"
                        if is_defective
                        else
                        "GOOD"
                    )


                    st.markdown(
                        f":{color}[{verdict}]"
                    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()