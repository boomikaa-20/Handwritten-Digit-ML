"""Streamlit website for the Handwritten Digit Recogniser.
Run with: streamlit run app.py
"""

import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import digit_recognizer as dr

st.set_page_config(
    page_title="Handwritten Digit Recogniser",
    page_icon="✏️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-digit {
        font-size: 110px;
        font-weight: 800;
        text-align: center;
        color: #1f77b4;
        line-height: 1.1;
        padding: 10px;
        border: 3px solid #1f77b4;
        border-radius: 16px;
        background: #f0f7ff;
    }
    .sub {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    """Load the trained model once and reuse it across reruns."""
    return dr.load_model()


@st.cache_data
def get_metrics():
    """Load evaluation metrics saved by train_model.py."""
    return dr.load_metrics()


# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "canvas_n" not in st.session_state:
    st.session_state.canvas_n = 0
if "last" not in st.session_state:
    st.session_state.last = None

# Sidebar navigation
st.sidebar.title("✏️ Digit Recogniser")
page = st.sidebar.radio(
    "Go to", ["Draw & Predict", "How It Works", "Model Evaluation"]
)
st.sidebar.markdown("---")
st.sidebar.caption("B.Tech Machine Learning Project\n\nMNIST • PCA + SVM • Streamlit")


def page_predict():
    st.title("✏️ Handwritten Digit Recogniser")
    st.write(
        "Draw **one digit (0–9)** with your mouse, then click **Predict**. "
        "A PCA + SVM model trained on the MNIST dataset recognises it."
    )

    try:
        model = get_model()
    except dr.ModelNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("1. Draw here")
        stroke = st.slider("Brush size", 10, 40, 25)
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=stroke,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=400,
            width=400,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=f"canvas_{st.session_state.canvas_n}",
        )
        button_col1, button_col2 = st.columns(2)
        predict_clicked = button_col1.button(
            "🔍 PREDICT", type="primary", use_container_width=True
        )
        clear_clicked = button_col2.button(
            "🧹 CLEAR CANVAS", use_container_width=True
        )

        # Download the ORIGINAL 400x400 canvas drawing (not the processed 28x28 image).
        if canvas.image_data is not None:
            drawing_array = np.asarray(canvas.image_data).astype(np.uint8)
            if drawing_array.ndim == 3 and drawing_array.shape[2] in (3, 4):
                drawing_image = Image.fromarray(drawing_array)
                drawing_buffer = io.BytesIO()
                drawing_image.save(drawing_buffer, format="PNG")
                st.download_button(
                    "⬇️ Download My Drawing (Input Image)",
                    data=drawing_buffer.getvalue(),
                    file_name="my_handwritten_digit.png",
                    mime="image/png",
                    key=f"download_input_{st.session_state.canvas_n}",
                    use_container_width=True,
                )

    if clear_clicked:
        st.session_state.canvas_n += 1
        st.session_state.last = None
        st.rerun()

    if predict_clicked:
        if canvas.image_data is None:
            st.warning("Draw a digit on the canvas first.")
        else:
            try:
                img28 = dr.preprocess_canvas(canvas.image_data)
                if img28 is None:
                    st.session_state.last = None
                    st.warning("The canvas looks blank. Please draw a digit first.")
                else:
                    result = dr.predict_digit(model, img28)
                    st.session_state.last = {"result": result, "img28": img28}
                    st.session_state.history.append(
                        {
                            "Time": time.strftime("%H:%M:%S"),
                            "Predicted digit": result["digit"],
                            "Top decision score": round(result["top3"][0][1], 3),
                            "Margin over 2nd": round(result["margin"], 3),
                        }
                    )
            except Exception as exc:
                st.error(f"Could not process this drawing: {exc}")

    with right:
        st.subheader("2. Result")
        last = st.session_state.last
        if last is None:
            st.info("Your prediction will appear here after you click Predict.")
        else:
            res = last["result"]
            img28 = last["img28"]

            st.markdown(
                f'<div class="big-digit">{res["digit"]}</div>'
                f'<div class="sub">Predicted Digit: {res["digit"]}</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            result_col, preview_col = st.columns([1, 1])

            with result_col:
                st.markdown("**Top 3 predictions**")
                top3 = pd.DataFrame(
                    {
                        "Rank": [1, 2, 3],
                        "Digit": [digit for digit, _score in res["top3"]],
                        "Decision score": [
                            round(score, 3) for _digit, score in res["top3"]
                        ],
                    }
                )
                st.dataframe(top3, hide_index=True, use_container_width=True)
                st.caption(
                    f"Margin over 2nd choice: {res['margin']:.3f} "
                    "(larger means the top score is further ahead)."
                )

            with preview_col:
                st.markdown("**What the model sees (28×28)**")
                preview = Image.fromarray(
                    (img28 * 255).astype(np.uint8)
                ).resize((196, 196), Image.Resampling.NEAREST)
                st.image(preview)

                # Download the processed image using Streamlit's supported control.
                image_buffer = io.BytesIO()
                preview.save(image_buffer, format="PNG")
                st.download_button(
                    label="⬇️ Download Processed Image",
                    data=image_buffer.getvalue(),
                    file_name=f"predicted_digit_{res['digit']}.png",
                    mime="image/png",
                    key="download_processed_image",
                )

    last = st.session_state.last
    if last is not None:
        st.subheader("3. Decision scores for all digits")
        scores = last["result"]["scores"]
        predicted_digit = last["result"]["digit"]
        colors = [
            "#1f77b4" if digit == predicted_digit else "#b0b0b0"
            for digit in range(10)
        ]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(10), scores, color=colors)
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit class")
        ax.set_ylabel("SVM decision score")
        ax.set_title("Higher score means the model favours that class; scores are not probabilities")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("📜 Prediction history", expanded=False):
        if st.session_state.history:
            st.dataframe(
                pd.DataFrame(st.session_state.history[::-1]),
                hide_index=True,
                use_container_width=True,
            )
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()
        else:
            st.write("No predictions yet.")


def page_how():
    st.title("🧠 How It Works")
    metrics = get_metrics()
    config = metrics["config"] if metrics else None

    st.markdown(
        """
### The pipeline
1. **Drawing** – draw on a 400×400 black canvas with a white brush.
2. **Preprocessing** – the app finds the digit's bounding box, crops it, scales its longest side to 20 pixels, places it in a 28×28 image, and centres it by centre of mass.
3. **Flattening** – the 28×28 image becomes 784 pixel values between 0 and 1.
4. **PCA** – compresses the 784 input features into fewer features.
5. **SVM** – classifies the processed image into one of ten classes (0–9).
6. **Output** – the predicted digit and the top-three decision scores are shown.

### Why MNIST?
MNIST contains 70,000 labelled 28×28 handwritten digit images: 60,000 training images and 10,000 test images in the standard split.

### What is PCA?
Principal Component Analysis reduces the number of input features by retaining directions with high variance. This can reduce computation while retaining useful information.

### What is an SVM?
A Support Vector Machine learns decision boundaries between classes. The RBF kernel can model non-linear boundaries. In this project, the trained model classifies images into digits 0–9.

### Decision scores vs probabilities
The chart displays SVM decision scores. These scores indicate which class the model favours, but they are **not probabilities** and should not be interpreted as percentages.
"""
    )

    if config:
        st.info(
            f"Configuration used: PCA with {config['pca_components']} components "
            f"(keeps {config['variance_kept'] * 100:.1f}% of variance), "
            f"SVM kernel={config['svm_kernel']}, C={config['svm_C']}, "
            f"gamma={config['svm_gamma']}, trained on {config['n_train']} images."
        )

    st.markdown(
        """
### Limitations
- Recognises one digit at a time.
- Mouse drawings can differ from MNIST handwriting, so live results may be less accurate than test-set results.
- Unusual writing styles may be misclassified.
- PCA + SVM is a classical ML approach; a CNN is a possible future enhancement.

### Future enhancements
A CNN, multi-digit recognition using segmentation, data augmentation, and evaluation on user-drawn samples could be explored in future work.
"""
    )


def page_eval():
    st.title("📊 Model Evaluation")
    metrics = get_metrics()
    if metrics is None:
        st.error("No metrics found. Please run: python train_model.py")
        st.stop()

    config = metrics["config"]
    svm = metrics["models"][metrics["primary_model"]]
    st.caption(
        f"Results from your training run: {config['n_train']} training images, "
        f"{config['n_test']} test images (MNIST standard split)."
    )

    st.subheader("Main model: PCA + SVM (RBF)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test accuracy", f"{svm['test_accuracy'] * 100:.2f}%")
    col2.metric("Precision (macro)", f"{svm['precision_macro'] * 100:.2f}%")
    col3.metric("Recall (macro)", f"{svm['recall_macro'] * 100:.2f}%")
    col4.metric("F1-score (macro)", f"{svm['f1_macro'] * 100:.2f}%")

    st.write(
        f"**Training accuracy** (measured on {config['train_accuracy_measured_on']} "
        f"training images): {svm['train_accuracy'] * 100:.2f}%  |  "
        f"**Test accuracy** (unseen data): {svm['test_accuracy'] * 100:.2f}%"
    )
    st.caption(
        "Test accuracy is the main measure of generalisation. Training accuracy "
        "is measured on data the model has already seen."
    )

    st.subheader("Model comparison")
    rows = []
    for name, model_metrics in metrics["models"].items():
        rows.append(
            {
                "Model": name,
                "Train accuracy (%)": round(model_metrics["train_accuracy"] * 100, 2),
                "Test accuracy (%)": round(model_metrics["test_accuracy"] * 100, 2),
                "Precision (macro, %)": round(model_metrics["precision_macro"] * 100, 2),
                "Recall (macro, %)": round(model_metrics["recall_macro"] * 100, 2),
                "F1 (macro, %)": round(model_metrics["f1_macro"] * 100, 2),
                "Training time (s)": round(model_metrics["train_time_sec"], 1),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Per-digit results (test set, PCA + SVM)")
    per_class = pd.DataFrame(metrics["per_class"])
    per_class.columns = ["Digit", "Precision", "Recall", "F1-score", "Test images"]
    st.dataframe(per_class.round(4), hide_index=True, use_container_width=True)

    st.subheader("Confusion matrix (test set, PCA + SVM)")
    confusion = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    threshold = confusion.max() / 2
    for row in range(10):
        for col in range(10):
            ax.text(
                col,
                row,
                int(confusion[row, col]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if confusion[row, col] > threshold else "black",
            )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Rows = true digits; columns = predicted digits. The diagonal shows correct "
        "predictions, while off-diagonal cells show misclassifications."
    )


if page == "Draw & Predict":
    page_predict()
elif page == "How It Works":
    page_how()
else:
    page_eval()
