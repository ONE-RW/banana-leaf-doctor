import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "banana_leaf_mobilenetv2.keras"
CLASS_PATH = APP_DIR / "class_names.json"
CONFIG_PATH = APP_DIR / "model_config.json"

IMG_SIZE = (224, 224)
LOW_CONFIDENCE = 0.60
MIN_PLANT_FRACTION = 0.30
DEFAULT_CLASSES = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]

INFO = {
    "healthy": {
        "title": "Healthy leaf",
        "color": "#2E9E5B",
        "status": "No disease detected",
        "about": "The leaf shows no sign of the diseases this model knows (cordana, sigatoka, pestalotiopsis).",
        "advice": [
            "Keep monitoring the plants regularly, especially after rain.",
            "Maintain good spacing, drainage and weed control.",
            "Remove old dry leaves to keep the plantation clean.",
        ],
    },
    "cordana": {
        "title": "Cordana leaf spot",
        "color": "#E8A317",
        "status": "Disease detected",
        "about": "A fungal leaf disease. It usually causes oval brown spots with a yellow edge, often starting at the leaf margin or tip.",
        "advice": [
            "Remove and destroy heavily infected leaves.",
            "Improve air flow and drainage; avoid overcrowded plants.",
            "Ask your local agricultural officer about suitable treatment if it spreads.",
        ],
    },
    "pestalotiopsis": {
        "title": "Pestalotiopsis leaf spot",
        "color": "#D9643A",
        "status": "Disease detected",
        "about": "A fungal disease that causes brown patches with darker borders. It often appears on leaves that are damaged or stressed.",
        "advice": [
            "Prune and remove affected leaf parts.",
            "Avoid wounding leaves and reduce plant stress (water and nutrients).",
            "Keep tools clean and consult an agricultural officer if it keeps spreading.",
        ],
    },
    "sigatoka": {
        "title": "Sigatoka leaf spot",
        "color": "#C0392B",
        "status": "Disease detected",
        "about": "A serious fungal disease of bananas. It starts as small yellow streaks that turn brown and dark, and it can destroy a lot of leaf area.",
        "advice": [
            "Remove infected leaves early and do not leave them in the field.",
            "Improve spacing, drainage and sunlight in the plantation.",
            "Contact an agricultural officer quickly to plan a control programme.",
        ],
    },
}

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding: 2rem 1.5rem 2.5rem; max-width: 1120px;}
.stApp {background: #F4F8F2;}
[data-testid="stSidebar"] {background: #EEF4EB;}
.eyebrow {text-transform: uppercase; letter-spacing: .12em; font-size: .76rem; font-weight: 700; color: #DDF2D9;}
.hero {
    background: linear-gradient(120deg, #174F36 0%, #247A4A 65%, #65A950 100%);
    padding: 2.4rem 2.5rem; border-radius: 24px; color: white; margin-bottom: 1.8rem;
    box-shadow: 0 12px 30px rgba(31,85,51,.16);
}
.hero h1 {margin: .35rem 0 0; font-size: clamp(2rem, 4vw, 2.7rem); color: white; letter-spacing: -.035em;}
.hero p {max-width: 720px; margin: .55rem 0 0; font-size: 1.04rem; opacity: .92; color: white; line-height: 1.6;}
.section-label {font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; color: #62806A; margin: .3rem 0 .2rem;}
.section-title {font-size: 1.35rem; font-weight: 700; color: #1F3525; margin: 0 0 1rem;}
.card {
    background: white; border: 1px solid #E4ECE1; border-radius: 20px; padding: 1.35rem 1.5rem;
    box-shadow: 0 8px 24px rgba(31,58,38,.045); margin-bottom: 1rem;
}
.card h3 {margin-top: 0;}
.result-card {
    background: white; border: 1px solid #E4ECE1; border-radius: 20px; padding: 1.3rem 1.5rem; margin-bottom: .7rem;
    box-shadow: 0 8px 24px rgba(31,58,38,.05);
}
.upload-card {background:white; border:1px solid #E4ECE1; border-radius:20px; padding:1.35rem 1.35rem .6rem; box-shadow:0 8px 24px rgba(31,58,38,.045);}
.upload-hint {font-size:.92rem; line-height:1.55; color:#627365; margin:.2rem 0 1rem;}
.chip {
    display: inline-block; color: white; padding: 0.25rem 0.8rem; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600; letter-spacing: 0.02em;
}
.result-title {font-size: 2rem; font-weight: 700; margin: 0.6rem 0 0.2rem 0;}
.result-conf {font-size: 1rem; color: #607064;}
.result-conf b {font-size: 1.35rem; color:#263B2B;}
.legend-chip {
    display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px; color: white;
    font-weight: 600; margin: 0.25rem 0.3rem 0.25rem 0; font-size: 0.9rem;
}
.disclaimer {font-size: 0.82rem; color: #6b7a6b; text-align: center; margin-top: 2rem; padding-top:1rem; border-top:1px solid #E1E9DE;}
[data-testid="stFileUploader"] {border-radius: 14px;}
[data-testid="stMetric"] {background:white; padding:.9rem 1rem; border:1px solid #E4ECE1; border-radius:14px;}
@media (max-width: 700px) { .block-container {padding:1rem .8rem 2rem;} .hero {padding:1.6rem 1.4rem; border-radius:18px;} }
</style>
"""


def show_plot(fig):
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def show_image(image, caption=None):
    try:
        st.image(image, caption=caption, width="stretch")
    except Exception:
        st.image(image, caption=caption, use_container_width=True)


@st.cache_resource(show_spinner="Loading the model...")
def load_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def load_class_names():
    if CLASS_PATH.exists():
        return json.loads(CLASS_PATH.read_text())
    return DEFAULT_CLASSES


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def preprocess(image):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = ImageOps.exif_transpose(image).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    return preprocess_input(arr)


def predict(model, image):
    return model.predict(preprocess(image), verbose=0)[0]


def info_for(name):
    return INFO.get(name, {"title": name.title(), "color": "#2E9E5B", "status": "Prediction",
                           "about": "", "advice": []})


def probability_chart(class_names, probs):
    order = np.argsort(probs)
    labels = [info_for(class_names[i])["title"] for i in order]
    values = [float(probs[i]) * 100 for i in order]
    colors = [info_for(class_names[i])["color"] for i in order]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        title="Probability for each class", height=300,
        margin=dict(l=10, r=40, t=50, b=30),
        xaxis=dict(range=[0, 108], title="Probability (%)", gridcolor="#E5ECE5"),
        yaxis=dict(automargin=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14, color="#1F2D1F"),
    )
    return fig


def confidence_gauge(confidence, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=confidence * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 44}},
        title={"text": "Model confidence", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#EEF3EE",
            "steps": [
                {"range": [0, 60], "color": "#F6DADA"},
                {"range": [60, 85], "color": "#F8EBC9"},
                {"range": [85, 100], "color": "#D8EFDF"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#1F2D1F"))
    return fig


def plant_fraction(image):
    img = ImageOps.exif_transpose(image).convert("RGB").resize((128, 128))
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    hue = hsv[..., 0] * 360.0 / 255.0
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    plant = (hue >= 35) & (hue <= 165) & (sat >= 0.18) & (val >= 0.15)
    return float(plant.mean())


def render_results(image):
    model = load_model(str(MODEL_PATH))
    class_names = load_class_names()

    fraction = plant_fraction(image)
    if fraction < MIN_PLANT_FRACTION:
        st.warning(
            f"This does not look like a banana leaf photo (only {fraction * 100:.0f}% of the image has "
            "green or leaf-like colours). The model only knows four classes, so it would still give an "
            "answer that means nothing. Please upload a close, clear photo of a banana leaf."
        )
        if not st.checkbox("Analyse anyway (my photo is a leaf)", key="analyze_anyway"):
            return None

    with st.spinner("Analysing the leaf..."):
        probs = predict(model, image)

    top = int(np.argmax(probs))
    name = class_names[top]
    confidence = float(probs[top])
    info = info_for(name)

    st.markdown(
        f"""
        <div class="result-card" style="border-left: 8px solid {info['color']};">
            <span class="chip" style="background:{info['color']};">{info['status']}</span>
            <div class="result-title">{info['title']}</div>
            <div class="result-conf"><b>{confidence * 100:.1f}%</b> confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence < LOW_CONFIDENCE:
        st.warning("The model is not very sure about this image. Try a clearer, closer photo of a single leaf "
                   "in good light. Photos that are not banana leaves can also give unreliable results.")

    col_a, col_b = st.columns([1, 1.4])
    with col_a:
        show_plot(confidence_gauge(confidence, info["color"]))
    with col_b:
        show_plot(probability_chart(class_names, probs))

    st.markdown(f"#### About: {info['title']}")
    st.write(info["about"])
    if info["advice"]:
        st.markdown("**What you can do**")
        for tip in info["advice"]:
            st.markdown(f"- {tip}")

    with st.expander("See all probabilities"):
        rows = sorted(zip(class_names, probs), key=lambda r: -r[1])
        lines = ["| Class | Probability |", "|---|---|"]
        lines += [f"| {info_for(c)['title']} | {p * 100:.2f}% |" for c, p in rows]
        st.markdown("\n".join(lines))

    return name, confidence


def get_image():
    source = st.radio("Photo source", ["Upload photo", "Use camera"], horizontal=True,
                      label_visibility="collapsed")
    if source == "Use camera":
        chosen = st.camera_input("Take a photo of the leaf")
    else:
        chosen = st.file_uploader("Drop a leaf photo here or browse files", type=["jpg", "jpeg", "png"],
                                  help="JPG or PNG · Use a clear, close photo in daylight")
    if chosen is None:
        return None
    try:
        return Image.open(chosen)
    except Exception:
        st.error("This file could not be read as an image. Please use a JPG or PNG photo.")
        return None


def render_sidebar(class_names):
    config = load_config()
    with st.sidebar:
        st.markdown("## About this app")
        st.write("This tool uses a deep learning model (MobileNetV2) to recognise banana leaf diseases from a photo.")
        st.markdown("**Classes**")
        for c in class_names:
            i = info_for(c)
            st.markdown(f"<span class='legend-chip' style='background:{i['color']}'>{i['title']}</span>",
                        unsafe_allow_html=True)
        if "test_accuracy" in config:
            st.metric("Test accuracy", f"{config['test_accuracy'] * 100:.1f}%")
        st.markdown("**Tips for a good photo**")
        st.markdown("- One leaf, filling most of the picture\n- Good daylight, no strong shadow\n"
                    "- Keep the leaf in focus\n- Avoid heavy filters")


def main():
    st.set_page_config(page_title="Banana Leaf Doctor", page_icon="🍌", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    class_names = load_class_names()
    render_sidebar(class_names)

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Banana crop health</div>
            <h1>🍃 Banana Leaf Doctor</h1>
            <p>A clear, simple check for common banana leaf diseases. Add a photo to see the most likely result and practical next steps.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: `{MODEL_PATH.name}`")
        st.info("Download `banana_leaf_mobilenetv2.keras` (and `class_names.json`) from your Kaggle notebook "
                "output and put them in the same folder as `app.py`, then refresh this page.")
        st.stop()

    left, right = st.columns([.9, 1.35], gap="large", vertical_alignment="top")

    with left:
        st.markdown('<div class="section-label">Step 1</div><div class="section-title">Add a leaf photo</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-card"><p class="upload-hint">Choose a clear photo with one leaf filling most of the frame.</p>', unsafe_allow_html=True)
        image = get_image()
        if image is not None:
            show_image(image, caption="Photo ready for analysis")
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("Photo tips"):
            st.markdown("- Use daylight and keep the leaf in focus\n- Avoid strong shadows and filters\n- Include one leaf, close up")

    with right:
        st.markdown('<div class="section-label">Step 2</div><div class="section-title">Leaf health report</div>', unsafe_allow_html=True)
        if image is None:
            st.markdown(
                """
                <div class="card">
                    <h3 style="margin-bottom:.35rem">Your report will appear here</h3>
                    <p style="color:#68786B;margin-top:0">Upload or take a photo to get a prediction and guidance.</p>
                    <p style="margin-bottom:.4rem"><b>Recognised conditions</b></p>
                    """
                + "".join(
                    f"<span class='legend-chip' style='background:{info_for(c)['color']}'>{info_for(c)['title']}</span>"
                    for c in class_names)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            render_results(image)

    st.markdown(
        "<div class='disclaimer'>This tool gives an AI estimate, not a diagnosis. "
        "Please confirm with an agricultural officer before applying any treatment.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
