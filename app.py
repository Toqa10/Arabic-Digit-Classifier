"""
🔢 Arabic Handwritten Digit Classifier — Streamlit App
=====================================================
المستخدم يرسم رقم بإيده → الموديل يتوقع الرقم إيه
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
import json, os, sys
sys.path.append('..')

# ── Arabic labels ──────────────────────────────────────────
ARABIC   = ['٠','١','٢','٣','٤','٥','٦','٧','٨','٩']
AR_NAMES = ['صفر','واحد','اتنين','تلاتة','أربعة',
            'خمسة','ستة','سبعة','تمانية','تسعة']
MEAN, STD = 0.1307, 0.3081

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title='Arabic Digit Classifier',
    page_icon='🔢',
    layout='wide'
)

# ── Model ──────────────────────────────────────────────────
class ArabicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    model = ArabicMLP()
    ckpt  = torch.load('../models/best.pth', map_location='cpu')
    model.load_state_dict(ckpt['state'])
    model.eval()
    return model, ckpt

def preprocess(img_array: np.ndarray) -> torch.Tensor:
    """
    بتاخد صورة من المستخدم وبتجهزها للموديل:
    1. Grayscale
    2. Resize 28×28
    3. Invert (خلفية بيضا → سودا)
    4. Normalize
    5. Flatten → Tensor
    """
    img = Image.fromarray(img_array).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype='float32') / 255.0
    arr = (arr - MEAN) / STD
    return torch.tensor(arr.flatten()).unsqueeze(0)  # (1, 784)

def predict(model, tensor):
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1).squeeze().numpy()
        pred  = probs.argmax()
    return int(pred), probs

# ── Load model ─────────────────────────────────────────────
try:
    model, ckpt = load_model()
    model_loaded = True
except:
    model_loaded = False

# ── UI ─────────────────────────────────────────────────────
st.title('🔢 Arabic Handwritten Digit Classifier')
st.markdown("""
**ارسم رقم عربي (٠–٩) في الكانفاس وهتوقعه الموديل بإيده**

الموديل عمل **98.42% accuracy** على 10,000 صورة test — مبني بـ PyTorch MLP من الصفر.
""")

col1, col2, col3 = st.columns([1.2, 1, 1.5])

with col1:
    st.subheader('🎨 ارسم هنا')

    # Drawing canvas
    try:
        from streamlit_drawable_canvas import st_canvas
        canvas = st_canvas(
            fill_color   = 'rgba(0,0,0,0)',
            stroke_width = 25,
            stroke_color = '#FFFFFF',
            background_color = '#000000',
            height       = 280,
            width        = 280,
            drawing_mode = 'freedraw',
            key          = 'canvas',
        )
        has_canvas = True
    except ImportError:
        st.info('نزّلي streamlit-drawable-canvas:\npip install streamlit-drawable-canvas')
        has_canvas = False

    if st.button('🗑️ مسح', use_container_width=True):
        st.rerun()

with col2:
    st.subheader('🎯 النتيجة')
    if model_loaded and has_canvas and canvas.image_data is not None:
        img_data = canvas.image_data.astype('uint8')
        if img_data.sum() > 1000:   # فيه رسم فعلي
            tensor = preprocess(img_data[:,:,0])  # channel أول
            pred, probs = predict(model, tensor)

            st.markdown(f"""
            <div style="text-align:center; padding:20px;
                        background:linear-gradient(135deg,#1a1a2e,#16213e);
                        border-radius:16px; margin:10px 0">
              <div style="font-size:80px; line-height:1">{ARABIC[pred]}</div>
              <div style="font-size:28px; color:#00d4ff; font-weight:700">{pred}</div>
              <div style="font-size:18px; color:#aaa; margin-top:8px">{AR_NAMES[pred]}</div>
              <div style="font-size:14px; color:#7fff7f; margin-top:10px">
                ثقة: {probs[pred]*100:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

            # top 3 predictions
            st.markdown('**Top 3 توقعات:**')
            top3 = probs.argsort()[::-1][:3]
            for i, idx in enumerate(top3):
                medal = ['🥇','🥈','🥉'][i]
                st.progress(float(probs[idx]),
                            text=f"{medal} {ARABIC[idx]} ({idx}) — {probs[idx]*100:.1f}%")
        else:
            st.info('ارسمي رقم في الكانفاس على الشمال')
    elif not model_loaded:
        st.error('الموديل مش موجود — شغّلي train.py أولاً')
    else:
        st.info('ارسمي رقم في الكانفاس على الشمال')

with col3:
    st.subheader('📊 كل الاحتمالات')
    if model_loaded and has_canvas and canvas.image_data is not None:
        img_data = canvas.image_data.astype('uint8')
        if img_data.sum() > 1000:
            tensor = preprocess(img_data[:,:,0])
            pred, probs = predict(model, tensor)
            import pandas as pd
            df = pd.DataFrame({
                'الرقم': [f'{ARABIC[i]} ({i})' for i in range(10)],
                'الاحتمال': probs,
                'نسبة مئوية': [f'{p*100:.2f}%' for p in probs]
            }).sort_values('الاحتمال', ascending=False)
            st.dataframe(df[['الرقم','نسبة مئوية']].reset_index(drop=True),
                         use_container_width=True, height=370)

# ── Model Info ─────────────────────────────────────────────
st.divider()
c1, c2, c3, c4 = st.columns(4)
if model_loaded:
    with c1: st.metric('🎯 Test Accuracy', f"{ckpt.get('acc',98.42):.2f}%")
with c2: st.metric('📦 Parameters',   '569,226')
with c3: st.metric('🏗️ Architecture', '784→512→256→128→10')
with c4: st.metric('⚡ Framework',    'PyTorch')

with st.expander('🧠 إزاي الموديل شغال؟'):
    st.markdown("""
    ### Architecture
    ```
    Input:    784 neurons  (28×28 pixels)
       ↓
    Layer 1:  512 neurons  + BatchNorm + ReLU + Dropout(0.3)
       ↓
    Layer 2:  256 neurons  + BatchNorm + ReLU + Dropout(0.3)
       ↓
    Layer 3:  128 neurons  + BatchNorm + ReLU + Dropout(0.15)
       ↓
    Output:   10 neurons   → Softmax → احتمالات
    ```

    ### إيه اللي بيحصل لما بترسمي رقم؟
    1. **الصورة** (280×280) بتتحول لـ (28×28) → 784 رقم
    2. كل رقم بيتضرب في **weight** معين
    3. بيعدي على **ReLU** عشان يتعلم patterns معقدة
    4. الـ **output** بيطلع 10 احتمالات
    5. أكبر احتمال = الرقم المتوقع

    ### نتائج التدريب
    - Training على **54,000** صورة
    - Validation على **6,000** صورة
    - Test على **10,000** صورة
    - **15 epochs** مع Adam optimizer
    """)
