# imports
import base64
import os
import tempfile
import datetime
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import plotly.graph_objects as go
from video_utils import extract_frames_smart, predict_video
from preprocessing import predict_with_tta, recommend_model, validate_image
from gradcam import get_gradcam_for_image
from report_gen import generate_pdf_report

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Deepfake Detection System", page_icon="🛡️", layout="wide")

# Load CSS
def load_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_local_css("./styles/style.css")

# ============================================================
# MODEL LOADING (unchanged logic)
# ============================================================
@st.cache_resource
def load_models():
    eff_net_model = tf.keras.models.load_model('EfficientNet_Models/efficientnetb3_binary_classifier_8.h5')
    eff_net_art_model = tf.keras.models.load_model('EfficientNet_Models/EfficientNet_fine_tune_art_model.h5')
    cnn_model = 'CNN_model_weight/model_weights.weights.h5'
    return eff_net_model, eff_net_art_model, cnn_model

eff_net_model, eff_net_art_model, cnn_model = load_models()

# ============================================================
# MODEL INFERENCE FUNCTIONS (unchanged)
# ============================================================
def run_cnn(img_arr):
    my_model = Sequential()
    my_model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=(256,256,3)))
    my_model.add(BatchNormalization()); my_model.add(MaxPooling2D())
    my_model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    my_model.add(BatchNormalization()); my_model.add(MaxPooling2D())
    my_model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    my_model.add(BatchNormalization()); my_model.add(MaxPooling2D())
    my_model.add(Flatten()); my_model.add(Dense(512, activation='relu'))
    my_model.add(Dropout(0.09)); my_model.add(Dense(1, activation='sigmoid'))
    my_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    my_model.load_weights(cnn_model)
    return my_model.predict(img_arr)

def run_effNet(img_arr):
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except ValueError:
        strategy = tf.distribute.get_strategy()
    with strategy.scope():
        return eff_net_model.predict(img_arr)

def run_effNet_Art(img_arr):
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except ValueError:
        strategy = tf.distribute.get_strategy()
    with strategy.scope():
        return eff_net_art_model.predict(img_arr)

# Preprocessing functions (unchanged)
def pre_process_img_effNet(image):
    img = load_img(image, target_size=(300, 300))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    return run_effNet(img_arr)

def pre_process_img_effNetArt(image):
    img = load_img(image, target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    return run_effNet_Art(img_arr)

def pre_process_img(image):
    input_picture = load_img(image, target_size=(256, 256))
    img_arr = img_to_array(input_picture) / 255.0
    img_arr = img_arr.reshape((1, 256, 256, 3))
    return run_cnn(img_arr)

RUN_MODEL_FNS = {'CNN': run_cnn, 'Efficientnet': run_effNet, 'Efficientnet Art': run_effNet_Art}
PREPROCESS_FUNCTIONS = {'CNN': pre_process_img, 'Efficientnet': pre_process_img_effNet, 'Efficientnet Art': pre_process_img_effNetArt}

# Category → Model mapping
CATEGORY_MAP = {
    "General Photo": "CNN",
    "AI Face": "Efficientnet",
    "Artwork / Anime": "Efficientnet Art",
}

# ============================================================
# SESSION STATE INIT
# ============================================================
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []
if "prev_image" not in st.session_state:
    st.session_state.prev_image = None
if "model_key" not in st.session_state:
    st.session_state.model_key = "default_model_key"
if "page" not in st.session_state:
    st.session_state.page = "🏠 Dashboard"
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "img_scan_triggered" not in st.session_state:
    st.session_state.img_scan_triggered = False
if "vid_scan_triggered" not in st.session_state:
    st.session_state.vid_scan_triggered = False

def add_to_history(filename, prediction, confidence, model_used, media_type="Image"):
    st.session_state.detection_history.insert(0, {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename[:30],
        "prediction": prediction,
        "confidence": f"{confidence:.2%}",
        "model": model_used,
        "type": media_type,
    })
    if len(st.session_state.detection_history) > 50:
        st.session_state.detection_history = st.session_state.detection_history[:50]

# ============================================================
# THEME CSS INJECTION
# ============================================================
if st.session_state.theme == "light":
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #f4f6fa !important;
        color: #1a1a2e !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8ecf1 0%, #dce1e8 100%) !important;
        border-right: 1px solid rgba(60, 70, 100, 0.18);
    }
    [data-testid="stSidebar"] * { color: #22273a !important; }
    [data-testid="stSidebar"] .stRadio label:hover { background: rgba(26,115,232,0.08) !important; }
    [data-testid="stSidebar"] .stRadio label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"] {
        background: rgba(26,115,232,0.12) !important;
        border-left: 3px solid #1a73e8 !important;
    }
    .page-title {
        background: linear-gradient(135deg, #1a73e8 0%, #6c63ff 50%, #e91e63 100%) !important;
        -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    }
    .page-subtitle { color: #555 !important; }
    .feature-card, .stat-card, .result-card {
        background: #ffffff !important;
        border: 1px solid rgba(60, 70, 100, 0.15) !important;
        color: #1a1a2e !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }
    .feature-card .card-title, .result-label { color: #1a1a2e !important; }
    .feature-card .card-desc { color: #555 !important; }
    .feature-card:hover { border-color: rgba(26,115,232,0.35) !important; box-shadow: 0 4px 16px rgba(26,115,232,0.08) !important; }
    .stat-card .stat-val { color: #1a73e8 !important; }
    .stat-card .stat-lbl { color: #555 !important; }
    .section-header { color: #1a73e8 !important; border-bottom-color: rgba(26,115,232,0.2) !important; }
    .gradcam-label { color: #1a73e8 !important; }
    .history-table { border-color: rgba(60,70,100,0.15) !important; }
    .history-table th { background: rgba(26,115,232,0.06) !important; color: #1a73e8 !important; border-bottom-color: rgba(26,115,232,0.15) !important; }
    .history-table td { color: #22273a !important; border-bottom-color: rgba(60,70,100,0.08) !important; }
    .history-table tr:hover td { background: rgba(26,115,232,0.03) !important; }
    .app-footer { color: #777 !important; border-top-color: rgba(0,0,0,0.08) !important; }
    .sidebar-footer { background: rgba(228,232,238,0.95) !important; color: #777 !important; border-top-color: rgba(0,0,0,0.06) !important; }
    .custom-hr { background: linear-gradient(90deg, transparent, rgba(26,115,232,0.2), transparent) !important; }
    .timeline-block { background: rgba(233,30,99,0.05) !important; color: #22273a !important; border-color: rgba(233,30,99,0.2) !important; }
    .timeline-block .timeline-title { color: #d81b60 !important; }
    .timeline-block .timeline-range { color: #22273a !important; }
    .result-verdict.real { background: linear-gradient(135deg, #00897b 0%, #1a73e8 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
    .result-verdict.fake { background: linear-gradient(135deg, #d32f2f 0%, #e91e63 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
    .stCheckbox label { color: #22273a !important; }
    [data-testid="stFileUploader"] section { background-color: #ffffff !important; border: 1px dashed rgba(26,115,232,0.4) !important; }
    [data-testid="stFileUploader"] small { color: #666666 !important; font-weight: 500 !important; }
    [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] p { color: #22273a !important; }
    [data-testid="stFileUploader"] button { background-color: #f4f6fa !important; color: #1a73e8 !important; border: 1px solid #1a73e8 !important; border-radius: 4px; padding: 0.25rem 0.75rem; }
    [data-testid="stFileUploader"] button:hover { background-color: #1a73e8 !important; color: #ffffff !important; }
    .stSelectbox div[data-baseweb="select"] > div { background-color: #ffffff !important; border-color: rgba(60,70,100,0.2) !important; }
    .stSelectbox label { color: #22273a !important; }
    .stSelectbox > div div { color: #22273a !important; }
    [data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"], li[role="option"] { background-color: #ffffff !important; }
    li[role="option"]:hover, [data-baseweb="menu"] > ul > li:hover { background-color: #f4f6fa !important; }
    [data-baseweb="popover"] *, [data-baseweb="menu"] * { color: #22273a !important; }
    .stButton > button { background-color: #ffffff !important; color: #22273a !important; border-color: rgba(60,70,100,0.2) !important; }
    .stDownloadButton > button { background-color: #ffffff !important; color: #22273a !important; border-color: rgba(60,70,100,0.2) !important; }
    [data-testid="stLinkButton"] > a { background-color: #ffffff !important; color: #22273a !important; border: 1px solid rgba(60,70,100,0.2) !important; text-decoration: none !important; }
    [data-testid="stLinkButton"] > a * { color: #22273a !important; }
    [data-testid="stLinkButton"] > a:hover { border-color: rgba(26,115,232,0.5) !important; color: #1a73e8 !important; }
    p, span, li, td, th, label, .stMarkdown { color: #22273a !important; }
    header[data-testid="stHeader"] { background: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
NAV_OPTIONS = ["🏠 Dashboard", "🖼️ Image Detection", "🎬 Video Detection", "📜 Detection History", "ℹ️ About Project"]

with st.sidebar:
    st.markdown("<p style='font-size:1.4rem;font-weight:700;color:#5af9fb;margin-bottom:0.2rem;'>🛡️ Deepfake Detection System</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem;color:#4a5568;margin-bottom:1.5rem;'>Media Authenticity Analyzer</p>", unsafe_allow_html=True)
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # Sync radio to session_state page
    current_index = NAV_OPTIONS.index(st.session_state.page) if st.session_state.page in NAV_OPTIONS else 0
    selected = st.radio(
        "Navigation", NAV_OPTIONS,
        index=current_index, label_visibility="collapsed",
        key="nav_radio"
    )
    # If user clicked a radio option, update session state
    if selected != st.session_state.page:
        st.session_state.page = selected
        st.rerun()

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # Theme toggle
    theme_label = "🌙 Dark Mode" if st.session_state.theme == "dark" else "☀️ Light Mode"
    if st.button("🌗 Toggle Theme", use_container_width=True, key="theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    st.caption(f"Current: {theme_label}")

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
    st.link_button("🚨 Report to Cybercell", "https://cybercrime.gov.in/", use_container_width=True)
    st.markdown(
        "<div class='sidebar-footer'>Developed for academics by<br>Ansh &middot; Aryan &middot; Kanishk</div>",
        unsafe_allow_html=True
    )

page = st.session_state.page


# ============================================================
# PAGE: DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    st.markdown("<p class='page-title'>Deepfake Detection System for Media Authenticity</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Detect AI-generated images and videos using explainable AI models</p>", unsafe_allow_html=True)

    # Animated eye (HTML/CSS/JS)
    eye_html = """
    <div style="display:flex;justify-content:center;margin:1.5rem 0;">
      <svg width="120" height="70" viewBox="0 0 120 70" id="cyberEye">
        <defs>
          <radialGradient id="irisGrad"><stop offset="0%" stop-color="#5af9fb"/><stop offset="100%" stop-color="#1a1a4e"/></radialGradient>
        </defs>
        <ellipse cx="60" cy="35" rx="55" ry="32" fill="none" stroke="#5af9fb" stroke-width="2" opacity="0.6"/>
        <ellipse cx="60" cy="35" rx="40" ry="25" fill="none" stroke="#5af9fb" stroke-width="1" opacity="0.3"/>
        <circle cx="60" cy="35" r="18" fill="url(#irisGrad)" opacity="0.8"/>
        <circle id="pupil" cx="60" cy="35" r="7" fill="#060918"/>
        <circle cx="55" cy="30" r="3" fill="rgba(255,255,255,0.3)"/>
      </svg>
    </div>
    <script>
    document.addEventListener('mousemove', function(e) {
      var p = document.getElementById('pupil');
      if (!p) return;
      var rect = p.ownerSVGElement.getBoundingClientRect();
      var cx = rect.left + rect.width/2, cy = rect.top + rect.height/2;
      var dx = e.clientX - cx, dy = e.clientY - cy;
      var dist = Math.sqrt(dx*dx + dy*dy);
      var max = 8;
      var r = Math.min(dist/8, max);
      var angle = Math.atan2(dy, dx);
      p.setAttribute('cx', 60 + r * Math.cos(angle));
      p.setAttribute('cy', 35 + r * Math.sin(angle));
    });
    </script>
    """
    import streamlit.components.v1 as components
    components.html(eye_html, height=100)

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # Feature cards — clickable navigation
    cards = [
        ("🖼️", "Image Detection", "Upload photos to detect AI-generated content with TTA-enhanced accuracy", "🖼️ Image Detection"),
        ("🎬", "Video Detection", "Analyze video frames with smart filtering and confidence-weighted decisions", "🎬 Video Detection"),
        ("🔥", "Grad-CAM Explainability", "Visualize which image regions influenced the model's prediction", "🖼️ Image Detection"),
        ("📊", "Confidence Scoring", "Calibrated probability scores with test-time augmentation averaging", "🖼️ Image Detection"),
        ("⏱️", "Timeline Detection", "Identify exact timestamps where deepfake segments appear in videos", "🎬 Video Detection"),
    ]
    cols = st.columns(5)
    for i, (icon, title, desc, target_page) in enumerate(cards):
        with cols[i]:
            st.markdown(
                f"<div class='feature-card'>"
                f"<span class='card-icon'>{icon}</span>"
                f"<div class='card-title'>{title}</div>"
                f"<div class='card-desc'>{desc}</div>"
                f"</div>", unsafe_allow_html=True
            )
            if st.button(f"Open {title}", key=f"card_{i}", use_container_width=True):
                st.session_state.page = target_page
                st.rerun()

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # Quick stats
    total = len(st.session_state.detection_history)
    fakes = sum(1 for h in st.session_state.detection_history if h["prediction"] == "AI Generated")
    reals = total - fakes
    c1, c2, c3 = st.columns(3)
    for col, val, label in [(c1, total, "Total Scans"), (c2, fakes, "Fakes Detected"), (c3, reals, "Real Verified")]:
        with col:
            st.markdown(f"<div class='stat-card'><span class='stat-val'>{val}</span><span class='stat-lbl'>{label}</span></div>", unsafe_allow_html=True)


# ============================================================
# PAGE: IMAGE DETECTION
# ============================================================
elif page == "🖼️ Image Detection":
    st.markdown("<p class='page-title'>🖼️ Image Detection</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Upload an image to analyze its authenticity</p>", unsafe_allow_html=True)
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        image_placeholder = st.empty()

    with right_col:
        user_image = st.file_uploader("Upload image", ['png', 'jpg', 'jpeg'], label_visibility='collapsed')

        if user_image:
            image_bytes = user_image.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_placeholder.markdown(
                f'<div style="display:flex;justify-content:center;"><img src="data:image/jpeg;base64,{image_base64}" style="max-width:100%;border-radius:12px;border:1px solid rgba(90,249,251,0.15);"/></div>',
                unsafe_allow_html=True
            )

        # Model recommendation
        if user_image is not None:
            user_image.seek(0)
            rec = recommend_model(user_image)
            st.markdown(f"<p style='color:#5af9fb;font-size:0.8rem;margin:0.3em 0;'>💡 Recommended: <b>{rec['recommended']}</b> — {rec['reason']}</p>", unsafe_allow_html=True)

        # Category selector
        category = st.selectbox("Select Input Type", list(CATEGORY_MAP.keys()), index=None, placeholder="Choose category...")
        manual_override = st.checkbox("🔧 Manual model override", value=False)
        if manual_override:
            model_name = st.selectbox("Select model", ['CNN', 'Efficientnet', 'Efficientnet Art'], index=None, placeholder="Choose model...")
        else:
            model_name = CATEGORY_MAP.get(category) if category else None

        # Scan button
        if user_image is not None and model_name is not None:
            if st.button("🔬 Run Scan", use_container_width=True, type="primary", key="img_scan_btn"):
                st.session_state.img_scan_triggered = True
        elif user_image is None or model_name is None:
            st.session_state.img_scan_triggered = False

        result_placeholder = st.empty()

    # Run prediction only after scan button click
    if user_image is not None and model_name is not None and st.session_state.img_scan_triggered:
        user_image.seek(0)
        try:
            tta_result = predict_with_tta(user_image, model_name, RUN_MODEL_FNS[model_name])
            if not tta_result['is_valid']:
                result_placeholder.error(f"⚠️ {tta_result['error']}")
            else:
                confidence = tta_result['confidence']
                result_word = tta_result['label']
                verdict_class = "real" if result_word == "REAL" else "fake"

                # Result card
                st.markdown(f"""
                <div class='result-card'>
                    <div class='result-label'>Prediction Result</div>
                    <div class='result-verdict {verdict_class}'>{result_word}</div>
                </div>""", unsafe_allow_html=True)

                # Stat cards
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(f"<div class='stat-card'><span class='stat-val'>{confidence:.2%}</span><span class='stat-lbl'>Confidence</span></div>", unsafe_allow_html=True)
                with s2:
                    st.markdown(f"<div class='stat-card'><span class='stat-val'>{model_name}</span><span class='stat-lbl'>Model Used</span></div>", unsafe_allow_html=True)
                with s3:
                    st.markdown(f"<div class='stat-card'><span class='stat-val'>4</span><span class='stat-lbl'>TTA Passes</span></div>", unsafe_allow_html=True)

                # Confidence stability meter
                tta_preds = tta_result.get('tta_predictions', [])
                if len(tta_preds) > 1:
                    spread = max(tta_preds) - min(tta_preds)
                    if spread < 0.05:
                        stability = "🟢 Very Stable"
                        stab_desc = "All augmentation passes agree closely."
                    elif spread < 0.15:
                        stability = "🟡 Moderately Stable"
                        stab_desc = "Minor variation across passes — result is reliable."
                    else:
                        stability = "🔴 Uncertain"
                        stab_desc = "Significant variation detected — interpret with caution."
                    st.markdown(f"<div class='stat-card' style='margin-top:0.8rem;'><span class='stat-val' style='font-size:1.1rem;'>{stability}</span><span class='stat-lbl'>Prediction Stability — {stab_desc}</span></div>", unsafe_allow_html=True)

                add_to_history(user_image.name, result_word, confidence, model_name, "Image")

                # Grad-CAM
                st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
                show_gradcam = st.checkbox("🔍 Show Grad-CAM Explanation", value=False, key='gradcam_toggle')
                gradcam_done = False
                if show_gradcam:
                    user_image.seek(0)
                    with st.spinner('Generating Grad-CAM heatmap...'):
                        models_dict = {'cnn_weights_path': cnn_model, 'eff_net_model': eff_net_model, 'eff_net_art_model': eff_net_art_model}
                        gradcam_result = get_gradcam_for_image(user_image, model_name, models_dict)
                    if gradcam_result['success']:
                        gradcam_done = True
                        if gradcam_result.get('used_fallback', False):
                            st.success("✅ Grad-CAM fallback layer applied successfully")
                        gc1, gc2 = st.columns(2)
                        with gc1:
                            st.markdown("<p class='gradcam-label'>Original</p>", unsafe_allow_html=True)
                            st.image(gradcam_result['original'], use_container_width=True)
                        with gc2:
                            st.markdown("<p class='gradcam-label'>Grad-CAM Heatmap</p>", unsafe_allow_html=True)
                            st.image(gradcam_result['overlay'], use_container_width=True)
                    else:
                        st.warning(f"⚠️ Grad-CAM generation failed: {gradcam_result['error']}")

                # --- Post-scan panels ---
                st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

                # Guidance panel
                if result_word != "REAL":
                    st.markdown("<p class='section-header'>📋 What Should You Do Next?</p>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class='timeline-block' style='border-left-color:#ff9800; border-color:rgba(255,152,0,0.25); background:rgba(255,152,0,0.06) !important;'>
                    <b>⚠️ This content may be AI-generated.</b><br>
                    • Do <b>not</b> forward or share this content without verification.<br>
                    • Try to verify the original source or contact the sender.<br>
                    • If this appears in a news or social media context, report it to the platform.<br>
                    • Consider filing a report at <a href='https://cybercrime.gov.in/' target='_blank' style='color:#5af9fb;'>cybercrime.gov.in</a> if the content is harmful.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("<p class='section-header'>✅ Assessment Complete</p>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class='timeline-block' style='border-left-color:#00c853; border-color:rgba(0,200,83,0.25); background:rgba(0,200,83,0.06) !important;'>
                    Content appears <b>authentic</b>. No immediate action required.<br>
                    If you still have concerns, consider cross-checking with other tools or sources.
                    </div>
                    """, unsafe_allow_html=True)

                # Report to Authority panel
                st.markdown("<p class='section-header'>🚨 Report Suspicious Content</p>", unsafe_allow_html=True)
                st.info("Redirecting to official reporting portal...", icon="ℹ️")
                ra1, ra2, ra3 = st.columns(3)
                with ra1:
                    st.link_button("🏛️ Report to Cybercell (Govt. of India)", "https://cybercrime.gov.in/", use_container_width=True)
                with ra2:
                    st.link_button("📢 Report to Platform (Meta / Instagram)", "https://www.facebook.com/help/contact/209046679279097", use_container_width=True)
                with ra3:
                    st.link_button("📞 Contact Local Authority (Police)", "https://services.india.gov.in/service/detail/lodge-complaint-with-police", use_container_width=True)

                # Download report
                st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
                pdf_buf = generate_pdf_report(
                    filename=user_image.name,
                    model_name=model_name,
                    label=result_word,
                    confidence=confidence,
                    media_type="Image",
                    category=category if category else None,
                    gradcam_available=gradcam_done,
                )
                st.download_button(
                    label="📄 Download Evidence Report (PDF)",
                    data=pdf_buf,
                    file_name=f"deepfake_report_{user_image.name.rsplit('.', 1)[0]}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="img_report_dl"
                )

        except Exception as e:
            result_placeholder.error(f"⚠️ Prediction failed: {str(e)}")


# ============================================================
# PAGE: VIDEO DETECTION
# ============================================================
elif page == "🎬 Video Detection":
    st.markdown("<p class='page-title'>🎬 Video Detection</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Upload a video to analyze frame-by-frame authenticity</p>", unsafe_allow_html=True)
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        video_placeholder = st.empty()

    with right_col:
        user_video = st.file_uploader("Upload video", ['mp4', 'avi', 'mov'], label_visibility='collapsed', key='video_uploader')
        category = st.selectbox("Select Input Type", list(CATEGORY_MAP.keys()), index=None, placeholder="Choose category...", key='vid_cat')
        manual_override = st.checkbox("🔧 Manual model override", value=False, key='vid_override')
        if manual_override:
            video_model_name = st.selectbox("Select model", ['CNN', 'Efficientnet', 'Efficientnet Art'], index=None, placeholder="Choose model...", key='vid_model')
        else:
            video_model_name = CATEGORY_MAP.get(category) if category else None

        # Scan button
        if user_video is not None and video_model_name is not None:
            if st.button("🔬 Run Scan", use_container_width=True, type="primary", key="vid_scan_btn"):
                st.session_state.vid_scan_triggered = True
        elif user_video is None or video_model_name is None:
            st.session_state.vid_scan_triggered = False

    if user_video is not None:
        video_placeholder.video(user_video)

    if user_video is not None and video_model_name is not None and st.session_state.vid_scan_triggered:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(user_video.read())
            tmp_video_path = tmp_file.name

        try:
            with st.spinner('Extracting and filtering frames...'):
                frames, skipped_blurry, skipped_duplicate, used_fallback = extract_frames_smart(tmp_video_path, frame_interval=5)

            st.markdown(
                f"<p style='color:#5af9fb;font-size:0.75rem;margin:0.5em 0;'>"
                f"📊 {len(frames)} frames extracted{' (fallback mode)' if used_fallback else ''}"
                f" | {skipped_blurry} blurry skipped | {skipped_duplicate} duplicate skipped</p>",
                unsafe_allow_html=True
            )

            if len(frames) == 0:
                st.error("Could not extract any frames. The video file may be corrupted.")
            else:
                progress_bar = st.progress(0, text="Analyzing frames...")
                def update_progress(current, total):
                    progress_bar.progress(current / total, text=f"Analyzing frame {current}/{total}...")

                video_results = predict_video(frames, video_model_name, PREPROCESS_FUNCTIONS,
                    progress_callback=update_progress, use_tta=False, smoothing_window=5)
                progress_bar.empty()

                result_word = video_results['final_label']
                fake_pct = video_results['fake_percentage']
                frame_results = video_results.get('frame_results', [])
                confs = [r['confidence'] for r in frame_results]
                avg_c, max_c, min_c = (np.mean(confs), np.max(confs), np.min(confs)) if confs else (0.5, 0.5, 0.5)

                verdict_class = "real" if result_word == "REAL" else "fake"
                st.markdown(f"<div class='result-card'><div class='result-label'>Video Verdict</div><div class='result-verdict {verdict_class}'>{result_word}</div></div>", unsafe_allow_html=True)

                # Stat cards
                cols = st.columns(5)
                stats = [(video_results['total_frames'], "Frames"), (video_results['fake_count'], "Fake"),
                         (video_results['real_count'], "Real"), (f"{fake_pct:.1f}%", "Fake %"), (f"{avg_c:.2%}", "Avg Conf")]
                for col, (val, lbl) in zip(cols, stats):
                    with col:
                        st.markdown(f"<div class='stat-card'><span class='stat-val'>{val}</span><span class='stat-lbl'>{lbl}</span></div>", unsafe_allow_html=True)

                # Confidence stability meter (video: based on frame consistency)
                if len(confs) > 1:
                    spread = max_c - min_c
                    std_dev = float(np.std(confs))
                    if std_dev < 0.05:
                        stability = "🟢 Very Stable"
                        stab_desc = "Frame predictions are highly consistent."
                    elif std_dev < 0.15:
                        stability = "🟡 Moderately Stable"
                        stab_desc = "Some variation across frames — result is reliable."
                    else:
                        stability = "🔴 Uncertain"
                        stab_desc = "High variation across frames — interpret with caution."
                    st.markdown(f"<div class='stat-card' style='margin-top:0.8rem;'><span class='stat-val' style='font-size:1.1rem;'>{stability}</span><span class='stat-lbl'>Prediction Stability — {stab_desc}</span></div>", unsafe_allow_html=True)

                add_to_history(user_video.name, result_word, avg_c, video_model_name, "Video")

                # Confidence graph
                st.markdown("<p class='section-header'>📈 Frame Confidence Timeline</p>", unsafe_allow_html=True)
                fig = go.Figure()
                frame_indices = list(range(len(confs)))
                fig.add_trace(go.Scatter(x=frame_indices, y=confs, mode='lines+markers',
                    line=dict(color='#5af9fb', width=2), marker=dict(size=4, color='#5af9fb'),
                    name='Confidence', hovertemplate='Frame %{x}<br>Confidence: %{y:.3f}'))
                fig.add_hline(y=0.5, line_dash="dash", line_color="#ff416c", opacity=0.6,
                    annotation_text="Threshold", annotation_position="top left")
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(6,9,24,0.8)',
                    font=dict(color='#7f8fa6', family='Inter'), height=300, margin=dict(l=40,r=20,t=30,b=40),
                    xaxis=dict(title="Frame #", gridcolor='rgba(90,249,251,0.06)'),
                    yaxis=dict(title="Confidence", range=[0,1], gridcolor='rgba(90,249,251,0.06)'),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Timeline detection: find fake segments
                fake_segments = []
                in_segment = False
                seg_start = None
                for r in frame_results:
                    if r['label'] == 'AI Generated':
                        if not in_segment:
                            seg_start = r.get('timestamp', 0)
                            in_segment = True
                    else:
                        if in_segment:
                            fake_segments.append((seg_start, r.get('timestamp', 0)))
                            in_segment = False
                if in_segment:
                    last_ts = frame_results[-1].get('timestamp', 0)
                    fake_segments.append((seg_start, last_ts))

                if fake_segments:
                    st.markdown("<p class='section-header'>⚠️ Fake Segment Timeline</p>", unsafe_allow_html=True)
                    for start, end in fake_segments:
                        st.markdown(
                            f"<div class='timeline-block'>"
                            f"<div class='timeline-title'>🔴 Fake detected</div>"
                            f"<div class='timeline-range'>{start:.1f}s — {end:.1f}s</div>"
                            f"</div>", unsafe_allow_html=True
                        )

                # --- Post-scan panels ---
                st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

                # Guidance panel
                if result_word != "REAL":
                    st.markdown("<p class='section-header'>📋 What Should You Do Next?</p>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class='timeline-block' style='border-left-color:#ff9800; border-color:rgba(255,152,0,0.25); background:rgba(255,152,0,0.06) !important;'>
                    <b>⚠️ This video may contain AI-generated content.</b><br>
                    • Do <b>not</b> forward or share this video without verification.<br>
                    • Check the original source and compare with trusted versions.<br>
                    • Report to the hosting platform if the content appears harmful.<br>
                    • File a complaint at <a href='https://cybercrime.gov.in/' target='_blank' style='color:#5af9fb;'>cybercrime.gov.in</a> if necessary.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("<p class='section-header'>✅ Assessment Complete</p>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class='timeline-block' style='border-left-color:#00c853; border-color:rgba(0,200,83,0.25); background:rgba(0,200,83,0.06) !important;'>
                    Video appears <b>authentic</b> across analyzed frames. No immediate action required.
                    </div>
                    """, unsafe_allow_html=True)

                # Report to Authority panel
                st.markdown("<p class='section-header'>🚨 Report Suspicious Content</p>", unsafe_allow_html=True)
                st.info("Redirecting to official reporting portal...", icon="ℹ️")
                ra1, ra2, ra3 = st.columns(3)
                with ra1:
                    st.link_button("🏛️ Report to Cybercell (Govt. of India)", "https://cybercrime.gov.in/", use_container_width=True)
                with ra2:
                    st.link_button("📢 Report to Platform (Meta / Instagram)", "https://www.facebook.com/help/contact/209046679279097", use_container_width=True)
                with ra3:
                    st.link_button("📞 Contact Local Authority (Police)", "https://services.india.gov.in/service/detail/lodge-complaint-with-police", use_container_width=True)

                # Download report
                st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
                pdf_buf = generate_pdf_report(
                    filename=user_video.name,
                    model_name=video_model_name,
                    label=result_word,
                    confidence=float(avg_c),
                    media_type="Video",
                    category=category if category else None,
                    gradcam_available=False,
                    video_summary=video_results,
                )
                st.download_button(
                    label="📄 Download Evidence Report (PDF)",
                    data=pdf_buf,
                    file_name=f"deepfake_report_{user_video.name.rsplit('.', 1)[0]}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="vid_report_dl"
                )

        except Exception as e:
            st.error(f"⚠️ Video analysis failed: {str(e)}")
        finally:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)


# ============================================================
# PAGE: DETECTION HISTORY
# ============================================================
elif page == "📜 Detection History":
    st.markdown("<p class='page-title'>📜 Detection History</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Recent detection results stored in this session</p>", unsafe_allow_html=True)
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    history = st.session_state.detection_history
    if not history:
        st.info("No detections yet. Upload an image or video to get started.")
    else:
        rows = "".join(
            f"<tr><td>{h['timestamp']}</td><td>{h['type']}</td><td>{h['filename']}</td>"
            f"<td>{h['prediction']}</td><td>{h['confidence']}</td><td>{h['model']}</td></tr>"
            for h in history
        )
        st.markdown(
            f"<table class='history-table'><thead><tr>"
            f"<th>Time</th><th>Type</th><th>File</th><th>Result</th><th>Confidence</th><th>Model</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>",
            unsafe_allow_html=True
        )
        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()


# ============================================================
# PAGE: ABOUT PROJECT
# ============================================================
elif page == "ℹ️ About Project":
    st.markdown("<p class='page-title'>ℹ️ About This Project</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Deepfake Detection System for Media Authenticity</p>", unsafe_allow_html=True)
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Objective
    Detect AI-generated images and videos using deep learning models with explainable AI techniques.

    ### 🧠 Models Used
    | Model | Input Type | Architecture |
    |---|---|---|
    | **CNN** | General photos | Custom 3-layer Conv2D + Dense |
    | **EfficientNet** | AI-generated faces | EfficientNetB3 (fine-tuned) |
    | **EfficientNet Art** | Artwork & anime | EfficientNet (fine-tuned on art) |

    ### 🔬 Key Features
    - **Test-Time Augmentation (TTA)** — 4 augmented passes averaged for stable predictions
    - **Grad-CAM Explainability** — Visual heatmap showing model attention regions
    - **Smart Video Analysis** — Blur/duplicate filtering, rolling-window smoothing
    - **Confidence-Weighted Decisions** — Mean confidence threshold for video verdicts
    - **Timeline Detection** — Identifies exact timestamps of fake segments

    ### 🛠️ Tech Stack
    TensorFlow, Keras, Streamlit, OpenCV, Plotly, PIL

    ### 📚 Academic Purpose
    This system was developed for academic deepfake detection research and demonstration.
    """)


# ============================================================
# FOOTER (all pages)
# ============================================================
st.markdown(
    "<div class='app-footer'>"
    "Deepfake Detection System for Media Authenticity<br>"
    "Built using TensorFlow, Streamlit and Explainable AI techniques<br>"
    "<span style='font-size:0.65rem;'>Developed for academics by Ansh &middot; Aryan &middot; Kanishk</span>"
    "</div>", unsafe_allow_html=True
)
