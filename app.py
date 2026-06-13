"""
🔢 Arabic Handwritten Digit Classifier — Streamlit App
Full redesign: dark theme, English UI, educational explanations
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import os

# ── Config ─────────────────────────────────────────────────
ARABIC    = ['٠','١','٢','٣','٤','٥','٦','٧','٨','٩']
EN_NAMES  = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']
MEAN, STD = 0.1307, 0.3081

st.set_page_config(
    page_title = "Digit Classifier · MLP Demo",
    page_icon  = "🔢",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Inject CSS (matches the dark purple design) ─────────────
st.markdown("""
<style>
  /* ── Base ─────────────────────────── */
  html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #f0f0f8 !important;
  }
  [data-testid="stAppViewContainer"] > .main { background: #0a0a0f; }
  [data-testid="block-container"] { padding: 2rem 2.5rem 4rem; }

  /* hide default header */
  [data-testid="stHeader"] { background: transparent; }

  /* ── Typography ───────────────────── */
  h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing:-0.5px; }
  .stMarkdown p { color: #9090b0; font-size: 14px; }

  /* ── Metric cards ─────────────────── */
  [data-testid="stMetric"] {
    background: #13131a;
    border: 1px solid #2a2a3a;
    border-radius: 14px;
    padding: 16px 20px !important;
  }
  [data-testid="stMetricLabel"] { color: #6b6b8a !important; font-size:11px !important; text-transform:uppercase; letter-spacing:.5px; }
  [data-testid="stMetricValue"] { color: #a78bfa !important; font-size:22px !important; font-weight:700 !important; }
  [data-testid="stMetricDelta"] { display:none; }

  /* ── Progress bar ─────────────────── */
  [data-testid="stProgress"] > div > div { background: #7c6af7 !important; border-radius:4px; }
  [data-testid="stProgress"] { background: #1c1c28 !important; border-radius:4px; }

  /* ── Expander ────────────────────── */
  [data-testid="stExpander"] {
    background: #13131a !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 14px !important;
  }
  [data-testid="stExpander"] summary { color: #a78bfa !important; font-weight:600; }

  /* ── Divider ─────────────────────── */
  hr { border-color: #2a2a3a !important; margin: 2rem 0 !important; }

  /* ── Dataframe ───────────────────── */
  [data-testid="stDataFrame"] { background: #13131a; border-radius:12px; }
  .stDataFrame th { background:#1c1c28 !important; color:#a78bfa !important; }
  .stDataFrame td { color:#f0f0f8 !important; }

  /* ── Canvas container ─────────────── */
  .canvas-container canvas {
    border-radius: 14px !important;
    border: 1.5px solid #2a2a3a !important;
  }

  /* ── Custom cards ─────────────────── */
  .info-card {
    background: #13131a;
    border: 1px solid #2a2a3a;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
  }
  .result-box {
    background: linear-gradient(135deg,#1a1828 0%,#110f1e 100%);
    border: 1px solid #7c6af7;
    border-radius: 18px;
    padding: 28px 20px 20px;
    text-align: center;
    box-shadow: 0 0 40px rgba(124,106,247,0.2);
    margin-bottom: 16px;
  }
  .step-card {
    background: #1c1c28;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 14px 16px;
    height: 100%;
  }
  .step-num {
    font-size: 11px; font-weight:700; color:#7c6af7;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;
  }
  .step-title { font-size:13px; font-weight:600; color:#f0f0f8; margin-bottom:4px; }
  .step-desc  { font-size:11px; color:#6b6b8a; line-height:1.5; }
  .why-card {
    background: #1c1c28;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 14px 16px;
  }
  .why-title { font-size:12px; font-weight:600; color:#a78bfa; margin-bottom:5px; }
  .why-desc  { font-size:11px; color:#6b6b8a; line-height:1.5; }
  .badge {
    display:inline-block;
    background: rgba(124,106,247,0.15);
    border: 1px solid rgba(124,106,247,0.35);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 11px;
    color: #a78bfa;
    font-weight:600;
    letter-spacing:.3px;
    margin-bottom:12px;
  }
  .arch-layer {
    background:#1c1c28;
    border:1px solid #2a2a3a;
    border-radius:10px;
    padding:10px 8px;
    text-align:center;
    flex:1;
  }
  .arch-n    { font-size:16px; font-weight:700; color:#a78bfa; }
  .arch-name { font-size:10px; color:#6b6b8a; margin-top:2px; }
  .arch-sub  { font-size:9px; color:#3a3a5a; margin-top:1px; font-style:italic; }
  .arch-row  { display:flex; align-items:center; gap:6px; }
  .arch-arr  { color:#3a3a5a; font-size:14px; flex-shrink:0; }
</style>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────
class ArabicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128,10),
        )
    def forward(self, x): return self.net(x)

@st.cache_resource
def load_model():
    # best.pth موجودة جنب app.py في نفس الـ folder
    app_dir    = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(app_dir, 'best.pth')
    model = ArabicMLP()
    ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state'])
    model.eval()
    return model, ckpt

def preprocess(img_array):
    img = Image.fromarray(img_array).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28,28), Image.LANCZOS)
    arr = np.array(img, dtype='float32') / 255.0
    arr = (arr - MEAN) / STD
    return torch.tensor(arr.flatten()).unsqueeze(0)

def predict(model, tensor):
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1).squeeze().numpy()
    return int(probs.argmax()), probs

try:
    model, ckpt = load_model()
    model_ok = True
except Exception as e:
    model_ok = False
    model_error = str(e)

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown('<div class="badge">● Live Demo &nbsp;·&nbsp; Phase 1 Project &nbsp;·&nbsp; PyTorch MLP</div>', unsafe_allow_html=True)
st.markdown("<h1 style='font-size:2.2rem;background:linear-gradient(135deg,#fff,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px'>Handwritten Digit Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#6b6b8a;font-size:14px;margin-bottom:28px'>A 3-layer neural network that reads your handwriting in real time. Draw any digit 0–9 and watch the model think through 10 possible answers simultaneously.</p>", unsafe_allow_html=True)

# ── Stats row ───────────────────────────────────────────────
m1,m2,m3,m4 = st.columns(4)
m1.metric("Test Accuracy",   "98.42%")
m2.metric("Parameters",      "569,226")
m3.metric("Training Images", "54,000")
m4.metric("Inference Time",  "< 5 ms")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MAIN — Canvas + Result + Probabilities
# ══════════════════════════════════════════════════════════
col_draw, col_result, col_probs = st.columns([1.1, 1, 1.1])

with col_draw:
    st.markdown("#### ✏️ Draw a digit")
    st.markdown("<p style='color:#6b6b8a;font-size:12px;margin-top:-8px;margin-bottom:10px'>Use your mouse or finger — draw clearly, centered in the box</p>", unsafe_allow_html=True)

    canvas_result = None
    try:
        from streamlit_drawable_canvas import st_canvas
        canvas_result = st_canvas(
            fill_color       = "rgba(0,0,0,0)",
            stroke_width     = 26,
            stroke_color     = "#FFFFFF",
            background_color = "#000000",
            height           = 300,
            width            = 300,
            drawing_mode     = "freedraw",
            key              = "main_canvas",
            display_toolbar  = False,
        )
        has_canvas = True
    except ImportError:
        st.error("Install: pip install streamlit-drawable-canvas")
        has_canvas = False

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑  Clear Canvas", use_container_width=True):
        st.rerun()

    st.markdown("""
    <div class="info-card">
      <div style="font-size:11px;color:#6b6b8a;line-height:1.7">
        <b style="color:#a78bfa">💡 Tips for best results</b><br>
        · Draw large, filling most of the box<br>
        · Keep the digit centered<br>
        · Use a single, confident stroke<br>
        · The model was trained on 28×28 images
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_result:
    st.markdown("#### 🎯 Prediction")

    if model_ok and has_canvas and canvas_result is not None and canvas_result.image_data is not None:
        img_data = canvas_result.image_data.astype('uint8')

        if img_data.sum() > 2000:
            tensor      = preprocess(img_data[:,:,0])
            pred, probs = predict(model, tensor)
            conf        = probs[pred] * 100

            # confidence colour
            conf_color = "#34d399" if conf >= 80 else "#fbbf24" if conf >= 50 else "#f87171"

            st.markdown(f"""
            <div class="result-box">
              <div style="font-size:90px;line-height:1">{ARABIC[pred]}</div>
              <div style="font-size:32px;font-weight:700;color:#a78bfa;margin-top:4px">{pred}</div>
              <div style="font-size:16px;color:#9090b0;margin-top:4px">{EN_NAMES[pred]}</div>
              <div style="margin-top:14px;display:inline-block;
                          background:rgba(52,211,153,0.12);
                          border:1px solid rgba(52,211,153,0.3);
                          border-radius:20px;padding:5px 16px;
                          font-size:13px;color:{conf_color};font-weight:600">
                ● {conf:.1f}% confident
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Top 3
            st.markdown("<p style='font-size:12px;color:#6b6b8a;margin-bottom:8px'>Top 3 candidates</p>", unsafe_allow_html=True)
            top3 = probs.argsort()[::-1][:3]
            medals = ["🥇","🥈","🥉"]
            for i, idx in enumerate(top3):
                st.progress(float(probs[idx]),
                            text=f"{medals[i]}  **{idx}** ({EN_NAMES[idx]})  —  {probs[idx]*100:.1f}%")

            # What changed vs blank
            st.markdown(f"""
            <div class="info-card" style="margin-top:14px">
              <div style="font-size:11px;color:#6b6b8a;line-height:1.7">
                <b style="color:#a78bfa">What the model saw</b><br>
                · Input: 300×300 → resized to 28×28<br>
                · Pixels normalized: (pixel÷255 − 0.1307) ÷ 0.3081<br>
                · Passed through 3 hidden layers<br>
                · Softmax turned 10 scores into probabilities
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:240px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;
                        color:#3a3a5a;font-size:14px;gap:10px">
              <div style="font-size:48px;opacity:.3">🧠</div>
              <div>Draw a digit to see the prediction</div>
            </div>
            """, unsafe_allow_html=True)

    elif not model_ok:
        st.error(f"Model error: {model_error}")
    else:
        st.markdown("""
        <div style="height:240px;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;
                    color:#3a3a5a;font-size:14px;gap:10px">
          <div style="font-size:48px;opacity:.3">🧠</div>
          <div>Draw a digit to see the prediction</div>
        </div>
        """, unsafe_allow_html=True)

with col_probs:
    st.markdown("#### 📊 All 10 probabilities")
    st.markdown("<p style='color:#6b6b8a;font-size:12px;margin-top:-8px;margin-bottom:12px'>Every digit has a probability — they sum to 100%</p>", unsafe_allow_html=True)

    if model_ok and has_canvas and canvas_result is not None and canvas_result.image_data is not None:
        img_data = canvas_result.image_data.astype('uint8')
        if img_data.sum() > 2000:
            tensor      = preprocess(img_data[:,:,0])
            pred, probs = predict(model, tensor)

            # Sort by probability
            order = probs.argsort()[::-1]
            for rank, idx in enumerate(order):
                pct     = probs[idx] * 100
                is_top  = (idx == pred)
                bar_col = "#7c6af7" if is_top else "#2a2a3a"
                txt_col = "#a78bfa" if is_top else "#6b6b8a"
                weight  = "700" if is_top else "400"
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:32px 1fr 44px;
                            align-items:center;gap:8px;margin-bottom:6px">
                  <div style="font-size:13px;font-weight:{weight};
                              color:{txt_col};text-align:center">{idx}</div>
                  <div style="height:8px;background:#1c1c28;border-radius:4px;overflow:hidden">
                    <div style="height:100%;width:{pct:.1f}%;
                                background:{bar_col};border-radius:4px;
                                transition:width .4s ease"></div>
                  </div>
                  <div style="font-size:11px;color:{txt_col};
                              font-weight:{weight};text-align:right">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-card" style="margin-top:10px">
              <div style="font-size:11px;color:#6b6b8a;line-height:1.7">
                <b style="color:#a78bfa">Why show all 10?</b><br>
                A confident model puts ~99% on one digit.
                If the top two are close (e.g. 3 vs 8), the
                drawing is ambiguous — the model is unsure, just like a human would be.
              </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════════════════
st.markdown("#### 🔬 How it works — from pixels to prediction")
st.markdown("<p style='color:#6b6b8a;font-size:13px;margin-bottom:18px'>Five steps happen in under 5 milliseconds every time you draw.</p>", unsafe_allow_html=True)

s1,s2,s3,s4,s5 = st.columns(5)
steps = [
    ("01","You draw","Your stroke is captured on a 300×300 pixel canvas"),
    ("02","Resize","Shrunk to 28×28 = 784 pixel values"),
    ("03","Normalize","Each pixel scaled: (pixel÷255 − mean) ÷ std"),
    ("04","3 layers","784 numbers flow through 569K weights and ReLU activations"),
    ("05","Softmax","10 raw scores become 10 probabilities that sum to 100%"),
]
for col, (num, title, desc) in zip([s1,s2,s3,s4,s5], steps):
    with col:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-num">Step {num}</div>
          <div class="step-title">{title}</div>
          <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════
st.markdown("#### 🏗️ Network architecture")
st.markdown("<p style='color:#6b6b8a;font-size:13px;margin-bottom:16px'>Each layer transforms the data into a progressively more abstract representation.</p>", unsafe_allow_html=True)

st.markdown("""
<div class="arch-row">
  <div class="arch-layer">
    <div class="arch-n">784</div>
    <div class="arch-name">Input</div>
    <div class="arch-sub">28×28 pixels</div>
  </div>
  <div class="arch-arr">→</div>
  <div class="arch-layer">
    <div class="arch-n">512</div>
    <div class="arch-name">Hidden 1</div>
    <div class="arch-sub">BatchNorm · ReLU · Dropout 30%</div>
  </div>
  <div class="arch-arr">→</div>
  <div class="arch-layer">
    <div class="arch-n">256</div>
    <div class="arch-name">Hidden 2</div>
    <div class="arch-sub">BatchNorm · ReLU · Dropout 30%</div>
  </div>
  <div class="arch-arr">→</div>
  <div class="arch-layer">
    <div class="arch-n">128</div>
    <div class="arch-name">Hidden 3</div>
    <div class="arch-sub">BatchNorm · ReLU · Dropout 15%</div>
  </div>
  <div class="arch-arr">→</div>
  <div class="arch-layer">
    <div class="arch-n">10</div>
    <div class="arch-name">Output</div>
    <div class="arch-sub">Softmax probabilities</div>
  </div>
</div>
<br>
""", unsafe_allow_html=True)

# Why each component
w1,w2,w3,w4 = st.columns(4)
why_cards = [
    ("⚡ ReLU Activation",
     "Without it, stacking layers is pointless — the whole network collapses to one straight line and can't learn curves or complex shapes."),
    ("📊 Batch Normalization",
     "Keeps each layer's outputs on a consistent scale. Without it, early layers dominate and training slows down or becomes unstable."),
    ("🎲 Dropout (30%)",
     "Randomly disables neurons during training. Forces the network to build redundant paths — so it still works on unseen handwriting."),
    ("🔢 Softmax Output",
     "Converts 10 raw scores into probabilities. Lets you see not just what the model predicts, but how confident it is in each option."),
]
for col, (title, desc) in zip([w1,w2,w3,w4], why_cards):
    with col:
        st.markdown(f"""
        <div class="why-card">
          <div class="why-title">{title}</div>
          <div class="why-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# TRAINING RESULTS (collapsible)
# ══════════════════════════════════════════════════════════
with st.expander("📈 Training details & per-class accuracy"):
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Training configuration**")
        cfg_df = pd.DataFrame({
            "Setting":  ["Optimizer","Loss","LR Scheduler","Batch Size","Epochs","Regularization","Augmentation"],
            "Value":    ["Adam (lr=0.001)","CrossEntropy + label smoothing 0.1",
                         "ReduceLROnPlateau","128","15",
                         "Dropout + Weight Decay 1e-4",
                         "Rotation ±15° + Affine"],
        })
        st.dataframe(cfg_df, use_container_width=True, hide_index=True)

    with t2:
        st.markdown("**Per-digit accuracy on 10,000 test images**")
        acc_df = pd.DataFrame({
            "Digit":    [f"{ARABIC[i]} ({i})" for i in range(10)],
            "Name":     EN_NAMES,
            "Accuracy": ["98.78%","99.47%","98.55%","98.22%","98.68%",
                         "98.21%","98.64%","98.05%","98.05%","97.42%"],
        })
        st.dataframe(acc_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **What these numbers mean**

    The model gets 98.42% of 10,000 test digits correct — that's 158 mistakes out of 10,000.
    The most common errors are between visually similar pairs: 4↔9, 3↔8, 5↔6.
    This matches how humans make mistakes on unclear handwriting too.
    """)

st.markdown("<br><p style='text-align:center;color:#3a3a5a;font-size:11px'>Built with PyTorch · Phase 1 of AI/ML Learning Roadmap · Trained on MNIST (60K images)</p>", unsafe_allow_html=True)
