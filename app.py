"""
app.py  —  Frontend Layer (Streamlit UI)
=========================================
Responsible for:
  - Page layout and styling
  - Collecting user inputs (sliders)
  - Calling model.py to get predictions and tips
  - Calling charts.py to get figures
  - Displaying everything to the user

This file has ZERO ML code and ZERO chart-drawing code.
It only imports from model.py and charts.py and wires everything together.
"""

import streamlit as st
import matplotlib.pyplot as plt

# ── Import our own modules ────────────────────────────────────────────────────
from model  import load_model, predict, generate_tips, DATASET_AVERAGES, PREMIUM_AVERAGES
from charts import make_gauge, make_comparison


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍷 Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── CSS styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
h1, h2, h3                  { font-family: 'Playfair Display', serif !important; }

.stApp {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a0a0a 50%, #0d0d0d 100%);
    color: #f0e6d3;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0505 0%, #2d0a0a 100%);
    border-right: 1px solid #4a1010;
}
div[data-testid="metric-container"] {
    background: rgba(120,20,20,0.2);
    border: 1px solid rgba(180,50,50,0.3);
    border-radius: 12px;
    padding: 16px;
}
.stButton > button {
    background: linear-gradient(135deg, #8b1a1a, #c0392b);
    color: white; border: none; border-radius: 8px;
    font-weight: 500; padding: 0.6rem 2rem;
}

/* Custom component classes */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; color: #d4a0a0;
    border-bottom: 1px solid rgba(180,50,50,0.3);
    padding-bottom: 8px; margin-bottom: 16px;
}
.quality-badge {
    display: inline-block; padding: 10px 24px;
    border-radius: 50px; font-family: 'Playfair Display', serif;
    font-size: 1.2rem; font-weight: 700;
}
.badge-premium { background: linear-gradient(135deg,#1a4a1a,#2d7a2d); border: 1px solid #4aaa4a; color: #90ff90; }
.badge-normal  { background: linear-gradient(135deg,#1a2a4a,#2d4a8a); border: 1px solid #4a7aaa; color: #90c0ff; }
.badge-low     { background: linear-gradient(135deg,#4a1a1a,#8a2d2d); border: 1px solid #aa4a4a; color: #ffb090; }
.insight-box   { background: rgba(139,26,26,0.15); border-left: 3px solid #8b1a1a; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-size: 0.88rem; color: #d4b8b8; }
.tip-box       { background: rgba(26,74,26,0.15);  border-left: 3px solid #2d7a2d; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-size: 0.88rem; color: #b8d4b8; }
.warn-box      { background: rgba(74,50,10,0.25);  border-left: 3px solid #c0820a; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-size: 0.88rem; color: #d4c4a0; }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

if model is None:
    st.error("❌ Place `winequality-white.csv` in the project folder and restart.")
    st.stop()


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem'>
    <h1 style='font-size:2.8rem; color:#d4a0a0; letter-spacing:2px; margin:0'>
        🍷 Wine Quality Predictor
    </h1>
    <p style='color:#8a6060; font-size:1rem; margin-top:6px; font-style:italic'>
        ML-powered analysis · UCI White Wine Dataset · Random Forest Model
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ── Sidebar — user inputs ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='section-header'>🧪 Wine Properties</div>", unsafe_allow_html=True)
    st.caption("Adjust sliders to describe your wine sample")

    st.markdown("**🏭 Acidity**")
    fixed_acidity    = st.slider("Fixed Acidity",    4.0,  14.0,   7.0,  0.1)
    volatile_acidity = st.slider("Volatile Acidity", 0.08,  1.1,   0.28, 0.01)
    citric_acid      = st.slider("Citric Acid",      0.0,   1.0,   0.34, 0.01)

    st.markdown("**🍬 Sugar & Salt**")
    residual_sugar   = st.slider("Residual Sugar",   0.6,  65.0,   5.0,  0.1)
    chlorides        = st.slider("Chlorides",        0.01,  0.35,  0.046, 0.001)

    st.markdown("**🧴 Sulfur Dioxide**")
    free_so2         = st.slider("Free SO₂",         1.0, 120.0,  35.0,  1.0)
    total_so2        = st.slider("Total SO₂",        9.0, 440.0, 138.0,  1.0)

    st.markdown("**⚗️ Physical**")
    density          = st.slider("Density",          0.987, 1.004,  0.994, 0.0001)
    ph               = st.slider("pH",               2.7,   4.0,    3.19,  0.01)
    sulphates        = st.slider("Sulphates",        0.2,   1.1,    0.49,  0.01)
    alcohol          = st.slider("Alcohol %",        8.0,  15.0,   10.5,   0.1)


# ── Build the inputs dict (passed to model and charts) ────────────────────────
inputs = {
    "fixed acidity":        fixed_acidity,
    "volatile acidity":     volatile_acidity,
    "citric acid":          citric_acid,
    "residual sugar":       residual_sugar,
    "chlorides":            chlorides,
    "free sulfur dioxide":  free_so2,
    "total sulfur dioxide": total_so2,
    "density":              density,
    "pH":                   ph,
    "sulphates":            sulphates,
    "alcohol":              alcohol,
}


# ── Get prediction and tips from model.py ─────────────────────────────────────
result     = predict(model, inputs)          # → {score, label, badge_class, emoji}
tip_result = generate_tips(inputs)           # → {tips, warnings}

score      = result["score"]
label      = result["label"]
badge      = result["badge_class"]
emoji      = result["emoji"]
tips       = tip_result["tips"]
warns      = tip_result["warnings"]


# ── Get charts from charts.py ─────────────────────────────────────────────────
gauge_fig   = make_gauge(score)
compare_fig = make_comparison(inputs, DATASET_AVERAGES, PREMIUM_AVERAGES)


# ── Layout — two columns ──────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("<div class='section-header'>📊 Prediction Result</div>", unsafe_allow_html=True)

    # Quality badge
    st.markdown(
        f"<div style='text-align:center; margin:12px 0'>"
        f"<span class='quality-badge {badge}'>{emoji} {label}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Gauge chart (from charts.py)
    st.pyplot(gauge_fig, use_container_width=True)
    plt.close(gauge_fig)

    # Key metric cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Score",      f"{score}/10")
    m2.metric("Alcohol",    f"{alcohol}%",
              delta=f"{alcohol - DATASET_AVERAGES['alcohol']:+.1f} vs avg")
    m3.metric("V. Acidity", f"{volatile_acidity}",
              delta=f"{volatile_acidity - DATASET_AVERAGES['volatile acidity']:+.3f} vs avg",
              delta_color="inverse")

with col_right:
    st.markdown("<div class='section-header'>📈 How Your Wine Compares</div>", unsafe_allow_html=True)

    # Comparison chart (from charts.py)
    st.pyplot(compare_fig, use_container_width=True)
    plt.close(compare_fig)


# ── Tips and warnings ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>💡 Smart Analysis & Tips</div>", unsafe_allow_html=True)

tip_col, warn_col = st.columns(2)

with tip_col:
    st.markdown("**Strengths**")
    if tips:
        for t in tips:
            st.markdown(f"<div class='tip-box'>{t}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='insight-box'>No standout strengths for this sample.</div>",
                    unsafe_allow_html=True)

with warn_col:
    st.markdown("**Areas to Improve**")
    if warns:
        for w in warns:
            st.markdown(f"<div class='warn-box'>{w}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='tip-box'>✅ No major quality issues detected!</div>",
                    unsafe_allow_html=True)


# ── Key insights panel ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>🔬 What Drives Wine Quality?</div>", unsafe_allow_html=True)

i1, i2, i3, i4 = st.columns(4)
i1.markdown("<div class='insight-box'>🥇 <b>Alcohol</b> — strongest predictor. Higher = higher score.</div>",      unsafe_allow_html=True)
i2.markdown("<div class='insight-box'>🥈 <b>Volatile Acidity</b> — keep it low. High = vinegar flavour.</div>",   unsafe_allow_html=True)
i3.markdown("<div class='insight-box'>🥉 <b>Density</b> — lighter wines tend to score higher.</div>",             unsafe_allow_html=True)
i4.markdown("<div class='insight-box'>4️⃣ <b>Residual Sugar</b> — moderate levels preferred.</div>",              unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#4a2020; font-size:0.8rem; padding:1rem 0'>
    Built with Python · Scikit-learn · Streamlit &nbsp;|&nbsp;
    UCI Wine Quality Dataset &nbsp;|&nbsp; Random Forest Regressor
</div>
""", unsafe_allow_html=True)
