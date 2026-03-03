import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ─────────────────────────────────────────────
st.set_page_config(page_title="🍷 Wine Quality Predictor", page_icon="🍷",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }
.stApp { background: linear-gradient(135deg, #0d0d0d 0%, #1a0a0a 50%, #0d0d0d 100%); color: #f0e6d3; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a0505 0%, #2d0a0a 100%); border-right: 1px solid #4a1010; }
div[data-testid="metric-container"] { background: rgba(120,20,20,0.2); border: 1px solid rgba(180,50,50,0.3); border-radius: 12px; padding: 16px; }
.stButton > button { background: linear-gradient(135deg, #8b1a1a, #c0392b); color: white; border: none; border-radius: 8px; font-weight: 500; padding: 0.6rem 2rem; }
.section-header { font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #d4a0a0; border-bottom: 1px solid rgba(180,50,50,0.3); padding-bottom: 8px; margin-bottom: 16px; }
.quality-badge { display: inline-block; padding: 10px 24px; border-radius: 50px; font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; }
.badge-premium { background: linear-gradient(135deg,#1a4a1a,#2d7a2d); border: 1px solid #4aaa4a; color: #90ff90; }
.badge-normal  { background: linear-gradient(135deg,#1a2a4a,#2d4a8a); border: 1px solid #4a7aaa; color: #90c0ff; }
.badge-low     { background: linear-gradient(135deg,#4a1a1a,#8a2d2d); border: 1px solid #aa4a4a; color: #ffb090; }
.insight-box { background: rgba(139,26,26,0.15); border-left: 3px solid #8b1a1a; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-size: 0.88rem; color: #d4b8b8; }
.tip-box  { background: rgba(26,74,26,0.15); border-left: 3px solid #2d7a2d; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-size: 0.88rem; color: #b8d4b8; }
.warn-box { background: rgba(74,50,10,0.25); border-left: 3px solid #c0820a; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-size: 0.88rem; color: #d4c4a0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
AVERAGES = {
    "fixed acidity":6.85,"volatile acidity":0.278,"citric acid":0.334,
    "residual sugar":6.39,"chlorides":0.046,"free sulfur dioxide":35.3,
    "total sulfur dioxide":138.4,"density":0.994,"pH":3.19,
    "sulphates":0.490,"alcohol":10.51
}
PREMIUM_AVG = {
    "fixed acidity":6.73,"volatile acidity":0.262,"citric acid":0.337,
    "residual sugar":5.18,"chlorides":0.043,"free sulfur dioxide":34.0,
    "total sulfur dioxide":125.1,"density":0.992,"pH":3.21,
    "sulphates":0.50,"alcohol":11.37
}

def engineer_features(data):
    df = data.copy()
    df['free_sulfur_ratio']   = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-5)
    df['acidity_balance']     = df['fixed acidity'] / (df['volatile acidity'] + 1e-5)
    df['sugar_alcohol_ratio'] = df['residual sugar'] / (df['alcohol'] + 1e-5)
    df['total_acidity']       = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
    return df

@st.cache_resource
def load_model():
    if os.path.exists("wine_quality_model_v2.pkl"):
        return joblib.load("wine_quality_model_v2.pkl")
    for csv in ["winequality-white.csv","winequality_white.csv"]:
        if os.path.exists(csv):
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            df = pd.read_csv(csv, sep=";").drop_duplicates()
            df = engineer_features(df)
            X, y = df.drop("quality",axis=1), df["quality"]
            pipe = Pipeline([("scaler",StandardScaler()),
                             ("model",RandomForestRegressor(n_estimators=200,max_depth=20,random_state=42,n_jobs=-1))])
            pipe.fit(X, y)
            joblib.dump(pipe,"wine_quality_model_v2.pkl")
            return pipe
    return None

@st.cache_data
def compute_model_metrics():
    """Train model and compute R², MAE, RMSE, and 5-fold cross-validation scores."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    for csv in ["winequality-white.csv","winequality_white.csv"]:
        if os.path.exists(csv):
            df = pd.read_csv(csv, sep=";").drop_duplicates()
            df = engineer_features(df)
            X, y = df.drop("quality", axis=1), df["quality"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # ── Random Forest (our model) ──
            rf_pipe = Pipeline([("scaler", StandardScaler()),
                                ("model", RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1))])
            rf_pipe.fit(X_train, y_train)
            y_pred = rf_pipe.predict(X_test)

            r2   = r2_score(y_test, y_pred)
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # 5-fold cross-validation
            cv_scores = cross_val_score(rf_pipe, X, y, cv=5, scoring="r2")

            # ── Baseline: Linear Regression (to show improvement) ──
            lr_pipe = Pipeline([("scaler", StandardScaler()),
                                ("model", LinearRegression())])
            lr_pipe.fit(X_train, y_train)
            lr_pred  = lr_pipe.predict(X_test)
            lr_r2    = r2_score(y_test, lr_pred)
            lr_mae   = mean_absolute_error(y_test, lr_pred)
            lr_rmse  = np.sqrt(mean_squared_error(y_test, lr_pred))

            return {
                "r2": r2, "mae": mae, "rmse": rmse,
                "cv_scores": cv_scores,
                "cv_mean": cv_scores.mean(),
                "cv_std":  cv_scores.std(),
                "lr_r2": lr_r2, "lr_mae": lr_mae, "lr_rmse": lr_rmse,
                "n_train": len(X_train), "n_test": len(X_test),
                "n_features": X.shape[1],
            }
    return None

def make_cv_chart(cv_scores, cv_mean):
    """Bar chart of 5 cross-validation fold scores."""
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#110505')

    fold_labels = [f"Fold {i+1}" for i in range(len(cv_scores))]
    colors = ['#2ecc71' if s >= cv_mean else '#c0392b' for s in cv_scores]
    bars = ax.bar(fold_labels, cv_scores, color=colors, alpha=0.85, edgecolor='none', width=0.5)

    # Mean line
    ax.axhline(cv_mean, color='#f0e6d3', linestyle='--', lw=1.5, label=f'Mean = {cv_mean:.3f}')

    for bar, val in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{val:.3f}', ha='center', fontsize=9, color='#d4b8b8')

    ax.set_ylim(0, 1)
    ax.set_ylabel("R² Score", color='#a08080', fontsize=9)
    ax.set_title("5-Fold Cross-Validation Results", color='#f0e6d3', fontfamily='serif', fontsize=11, pad=8)
    ax.legend(facecolor='#1a0505', labelcolor='#d4b8b8', fontsize=8, framealpha=0.8)
    ax.spines[:].set_color('#3a1010')
    ax.tick_params(colors='#a08080')
    ax.set_xticklabels(fold_labels, color='#d4b8b8', fontsize=9)
    plt.tight_layout()
    return fig

def make_model_comparison_chart(metrics):
    """Side-by-side bar chart comparing Random Forest vs Linear Regression."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.patch.set_facecolor('#0d0d0d')

    data = {
        "R² Score\n(higher = better)":  ([metrics["lr_r2"],  metrics["r2"]],   True),
        "MAE\n(lower = better)":         ([metrics["lr_mae"], metrics["mae"]],   False),
        "RMSE\n(lower = better)":        ([metrics["lr_rmse"],metrics["rmse"]],  False),
    }

    for ax, (title, (vals, higher_better)) in zip(axes, data.items()):
        ax.set_facecolor('#110505')
        # Green for better model, red for worse
        if higher_better:
            clrs = ['#555588', '#2ecc71']
        else:
            clrs = ['#555588', '#2ecc71']
        bars = ax.bar(["Linear\nRegression", "Random\nForest"], vals, color=clrs, alpha=0.85, edgecolor='none', width=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f'{val:.3f}', ha='center', fontsize=9, color='#f0e6d3', fontweight='bold')
        ax.set_title(title, color='#d4b8b8', fontsize=9, pad=6)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_yticks([])
        ax.spines[:].set_color('#3a1010')
        ax.tick_params(colors='#a08080')
        ax.set_xticklabels(["Linear\nRegression", "Random\nForest"], color='#d4b8b8', fontsize=8)

    fig.suptitle("Random Forest vs Linear Regression", color='#f0e6d3', fontfamily='serif', fontsize=11)
    plt.tight_layout()
    return fig

def make_gauge(score):
    fig, ax = plt.subplots(figsize=(5,3))
    fig.patch.set_facecolor('#0d0d0d'); ax.set_facecolor('#0d0d0d')
    bands = [(3,4.5,'#7f1d1d'),(4.5,6,'#1d3557'),(6,7.5,'#1a4a6e'),(7.5,9,'#145a32')]
    r_o, r_i = 1.0, 0.6
    for lo,hi,color in bands:
        t = np.linspace(np.pi-(lo-3)/6*np.pi, np.pi-(hi-3)/6*np.pi, 60)
        ax.fill_between(np.cos(t)*r_o, np.sin(t)*r_o, np.cos(t)*r_i, alpha=0.85, color=color)
    angle = np.pi - (min(max(score,3),9)-3)/6*np.pi
    ax.annotate("", xy=(np.cos(angle)*0.75, np.sin(angle)*0.75), xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color='#f0e6d3', lw=2, mutation_scale=15))
    ax.plot(0,0,'o',color='#c0392b',markersize=8,zorder=5)
    ax.text(0,0.15,f"{score:.1f}",ha='center',va='center',fontsize=22,fontweight='bold',color='#f0e6d3',fontfamily='serif')
    ax.text(0,-0.05,"Quality Score",ha='center',va='center',fontsize=9,color='#a08080')
    for val,lbl in [(3,'3\nLow'),(5.25,'5\nNormal'),(7.5,'7\nGood'),(9,'9\nPremium')]:
        a = np.pi-(val-3)/6*np.pi
        ax.text(np.cos(a)*1.12,np.sin(a)*1.12,lbl,ha='center',va='center',fontsize=7,color='#a08080')
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-0.3,1.3); ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def make_comparison(inputs):
    keys = ["alcohol","volatile acidity","residual sugar","pH","sulphates","chlorides"]
    labels = ["Alcohol","Volatile\nAcidity","Residual\nSugar","pH","Sulphates","Chlorides"]
    u = [inputs[k] for k in keys]
    a = [AVERAGES[k] for k in keys]
    p = [PREMIUM_AVG[k] for k in keys]
    mx = [max(ui,ai,pi) for ui,ai,pi in zip(u,a,p)]
    u_n=[v/m if m else 0 for v,m in zip(u,mx)]
    a_n=[v/m if m else 0 for v,m in zip(a,mx)]
    p_n=[v/m if m else 0 for v,m in zip(p,mx)]
    x=np.arange(len(labels)); w=0.25
    fig,ax=plt.subplots(figsize=(9,4))
    fig.patch.set_facecolor('#0d0d0d'); ax.set_facecolor('#110505')
    ax.bar(x-w,u_n,w,label='Your Wine',color='#c0392b',alpha=0.9)
    ax.bar(x,  a_n,w,label='Dataset Avg',color='#2471a3',alpha=0.7)
    ax.bar(x+w,p_n,w,label='Premium Avg',color='#1e8449',alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels,color='#d4b8b8',fontsize=9)
    ax.set_yticks([]); ax.set_title("Your Wine vs Average vs Premium",color='#f0e6d3',fontfamily='serif',fontsize=12,pad=10)
    ax.legend(facecolor='#1a0505',labelcolor='#d4b8b8',fontsize=8,framealpha=0.8)
    ax.spines[:].set_color('#3a1010'); ax.tick_params(colors='#a08080')
    plt.tight_layout(); return fig

def generate_tips(inputs):
    tips, warns = [], []
    if inputs["volatile acidity"] > 0.35: warns.append("⚠️ **Volatile acidity is high** — gives wine a vinegar taste. Keep below 0.30 for premium quality.")
    if inputs["alcohol"] < 10.0:          warns.append("⚠️ **Alcohol is low** — alcohol is the #1 quality driver. Premium wines average 11%+.")
    if inputs["residual sugar"] > 12:     warns.append("⚠️ **Residual sugar is very high** — may make wine overly sweet and reduce score.")
    if inputs["chlorides"] > 0.07:        warns.append("⚠️ **Chlorides elevated** — high salt content negatively impacts taste.")
    if inputs["alcohol"] >= 11.0:         tips.append("✅ **Good alcohol level** — within the premium wine range.")
    if inputs["volatile acidity"] <= 0.28:tips.append("✅ **Volatile acidity well-controlled** — minimal vinegar taste expected.")
    if 20 < inputs["free sulfur dioxide"] < 60: tips.append("✅ **SO₂ levels balanced** — good preservation without excess.")
    if 3.0 <= inputs["pH"] <= 3.4:        tips.append("✅ **pH in ideal range** — good acidity balance for white wine.")
    return tips, warns

# ─────────── HEADER ───────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem'>
<h1 style='font-size:2.8rem;color:#d4a0a0;letter-spacing:2px;margin:0'>🍷 Wine Quality Predictor</h1>
<p style='color:#8a6060;font-size:1rem;margin-top:6px;font-style:italic'>ML-powered analysis · UCI White Wine Dataset · Random Forest Model</p>
</div>""", unsafe_allow_html=True)
st.markdown("---")

model = load_model()
if model is None:
    st.error("❌ Place `winequality-white.csv` or `wine_quality_model_v2.pkl` in the same folder.")
    st.stop()

# ─────────── SIDEBAR ───────────
with st.sidebar:
    st.markdown("<div class='section-header'>🧪 Wine Properties</div>", unsafe_allow_html=True)
    st.caption("Adjust sliders to describe your wine sample")

    st.markdown("**🏭 Acidity**")
    fixed_acidity    = st.slider("Fixed Acidity",    4.0,14.0,7.0,0.1)
    volatile_acidity = st.slider("Volatile Acidity", 0.08,1.1,0.28,0.01)
    citric_acid      = st.slider("Citric Acid",      0.0,1.0,0.34,0.01)
    st.markdown("**🍬 Sugar & Salt**")
    residual_sugar   = st.slider("Residual Sugar",   0.6,65.0,5.0,0.1)
    chlorides        = st.slider("Chlorides",        0.01,0.35,0.046,0.001)
    st.markdown("**🧴 Sulfur Dioxide**")
    free_so2         = st.slider("Free SO₂",         1.0,120.0,35.0,1.0)
    total_so2        = st.slider("Total SO₂",        9.0,440.0,138.0,1.0)
    st.markdown("**⚗️ Physical**")
    density          = st.slider("Density",          0.987,1.004,0.994,0.0001)
    ph               = st.slider("pH",               2.7,4.0,3.19,0.01)
    sulphates        = st.slider("Sulphates",        0.2,1.1,0.49,0.01)
    alcohol          = st.slider("Alcohol %",        8.0,15.0,10.5,0.1)

inputs = {
    "fixed acidity":fixed_acidity,"volatile acidity":volatile_acidity,
    "citric acid":citric_acid,"residual sugar":residual_sugar,
    "chlorides":chlorides,"free sulfur dioxide":free_so2,
    "total sulfur dioxide":total_so2,"density":density,
    "pH":ph,"sulphates":sulphates,"alcohol":alcohol
}

score = float(model.predict(engineer_features(pd.DataFrame([inputs])))[0])
score = round(min(max(score,0),10),2)

if score >= 7:   label,badge,emoji = "Premium Quality","badge-premium","🏆"
elif score >= 5: label,badge,emoji = "Normal Quality", "badge-normal", "✅"
else:            label,badge,emoji = "Low Quality",    "badge-low",    "⚠️"

tips, warns = generate_tips(inputs)

# ─────────── RESULTS ───────────
c1, c2 = st.columns([1,1], gap="large")

with c1:
    st.markdown("<div class='section-header'>📊 Prediction Result</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;margin:12px 0'><span class='quality-badge {badge}'>{emoji} {label}</span></div>", unsafe_allow_html=True)
    st.pyplot(make_gauge(score), use_container_width=True); plt.close()
    m1,m2,m3 = st.columns(3)
    m1.metric("Score", f"{score}/10")
    m2.metric("Alcohol", f"{alcohol}%", delta=f"{alcohol-AVERAGES['alcohol']:+.1f} vs avg")
    m3.metric("V. Acidity", f"{volatile_acidity}", delta=f"{volatile_acidity-AVERAGES['volatile acidity']:+.3f} vs avg", delta_color="inverse")

with c2:
    st.markdown("<div class='section-header'>📈 How Your Wine Compares</div>", unsafe_allow_html=True)
    st.pyplot(make_comparison(inputs), use_container_width=True); plt.close()

st.markdown("---")
st.markdown("<div class='section-header'>💡 Smart Analysis & Tips</div>", unsafe_allow_html=True)
tc, wc = st.columns(2)
with tc:
    st.markdown("**Strengths**")
    if tips:
        for t in tips: st.markdown(f"<div class='tip-box'>{t}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='insight-box'>No standout strengths for this sample.</div>", unsafe_allow_html=True)
with wc:
    st.markdown("**Areas to Improve**")
    if warns:
        for w in warns: st.markdown(f"<div class='warn-box'>{w}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='tip-box'>✅ No major quality issues detected!</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div class='section-header'>🔬 What Drives Wine Quality?</div>", unsafe_allow_html=True)
i1,i2,i3,i4 = st.columns(4)
i1.markdown("<div class='insight-box'>🥇 <b>Alcohol</b> — strongest predictor. Higher alcohol = higher score.</div>", unsafe_allow_html=True)
i2.markdown("<div class='insight-box'>🥈 <b>Volatile Acidity</b> — keep it low. High = vinegar flavour.</div>", unsafe_allow_html=True)
i3.markdown("<div class='insight-box'>🥉 <b>Density</b> — lighter wines tend to score higher.</div>", unsafe_allow_html=True)
i4.markdown("<div class='insight-box'>4️⃣ <b>Residual Sugar</b> — moderate levels preferred.</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div class='section-header'>📐 Model Evaluation & Performance</div>", unsafe_allow_html=True)

with st.spinner("Computing model metrics..."):
    metrics = compute_model_metrics()

if metrics:
    with st.expander("💡 What do these metrics mean? (Click to read)", expanded=False):
        st.markdown("""
<div style='color:#d4b8b8; font-size:0.88rem; line-height:1.8'>
<b style='color:#d4a0a0'>R² Score</b> — How well the model explains quality variation. Range 0–1. R²=0.55 means 55% of what makes wines different in quality is captured by the model.<br><br>
<b style='color:#d4a0a0'>MAE (Mean Absolute Error)</b> — Average prediction error in quality points. MAE=0.48 means predictions are off by 0.48 points on average (scale 0–10). Lower = better.<br><br>
<b style='color:#d4a0a0'>RMSE (Root Mean Squared Error)</b> — Like MAE but punishes large errors more. If RMSE much bigger than MAE, the model makes occasional very wrong predictions. Lower = better.<br><br>
<b style='color:#d4a0a0'>5-Fold Cross-Validation</b> — Dataset split into 5 parts, model trained and tested 5 times on different splits. Consistent scores prove the model is reliable, not just lucky on one test.
</div>""", unsafe_allow_html=True)

    st.markdown("#### Our Model (Random Forest) — Test Set Results")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("R² Score",       f"{metrics['r2']:.4f}",  help="Closer to 1.0 = better.")
    mc2.metric("MAE",            f"{metrics['mae']:.4f}", help="Average error in quality points.")
    mc3.metric("RMSE",           f"{metrics['rmse']:.4f}",help="Penalises large errors more.")
    mc4.metric("CV R² (5-Fold)", f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}", help="Mean ± Std across 5 folds.")

    r2_imp  = ((metrics['r2']  - metrics['lr_r2'])  / abs(metrics['lr_r2']))  * 100
    mae_imp = ((metrics['lr_mae'] - metrics['mae']) / metrics['lr_mae']) * 100
    st.markdown(f"<div class='tip-box'>🏆 <b>Random Forest outperforms Linear Regression by {r2_imp:.1f}% in R²</b> and reduces prediction error (MAE) by {mae_imp:.1f}% — this is why we chose it as our final model.</div>", unsafe_allow_html=True)

    chart1, chart2 = st.columns([1, 1], gap="large")
    with chart1:
        st.markdown("**5-Fold Cross-Validation R² per Fold**")
        st.caption("Consistent bars across all 5 folds = model generalises well, not overfitting")
        cv_fig = make_cv_chart(metrics["cv_scores"], metrics["cv_mean"])
        st.pyplot(cv_fig, use_container_width=True); plt.close()
    with chart2:
        st.markdown("**Random Forest vs Linear Regression**")
        st.caption("Baseline comparison — shows the upgrade from simple to advanced model")
        comp_fig = make_model_comparison_chart(metrics)
        st.pyplot(comp_fig, use_container_width=True); plt.close()

    st.markdown("#### Dataset Summary")
    ds1, ds2, ds3, ds4 = st.columns(4)
    ds1.metric("Total Samples", "4,898")
    ds2.metric("Training Set",  f"{metrics['n_train']}")
    ds3.metric("Test Set",      f"{metrics['n_test']}")
    ds4.metric("Features Used", f"{metrics['n_features']} (incl. 4 engineered)")

st.markdown("""<div style='text-align:center;color:#4a2020;font-size:0.8rem;padding:1rem 0'>
Built with Python · Scikit-learn · Streamlit &nbsp;|&nbsp; UCI Wine Quality Dataset &nbsp;|&nbsp; Random Forest Regressor
</div>""", unsafe_allow_html=True)
