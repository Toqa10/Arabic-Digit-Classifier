# 🔢 Arabic Handwritten Digits Classifier

> **MLP بنيته من الصفر بـ PyTorch — 98.42% accuracy على 10,000 صورة test**

---

## 🎯 المشروع

تصنيف الأرقام المكتوبة بإيد (٠–٩) باستخدام **Multi-Layer Perceptron** مبني بـ PyTorch.

**التطبيقات الحقيقية:**
- 🏦 قراءة أرقام الشيكات في البنوك
- 📬 قراءة أكواد البريد على الخطابات
- 🏥 رقمنة الوصفات الطبية المكتوبة
- 📄 Document scanning systems

---

## 📊 النتائج

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **98.42%** |
| Test Loss | 0.5434 |
| Best Val Accuracy | 98.42% |
| Total Parameters | 569,226 |

### Per-class Accuracy
| الرقم | Accuracy |
|-------|----------|
| ٠ (0) | 98.78% |
| ١ (1) | 99.47% |
| ٢ (2) | 98.55% |
| ٣ (3) | 98.22% |
| ٤ (4) | 98.68% |
| ٥ (5) | 98.21% |
| ٦ (6) | 98.64% |
| ٧ (7) | 98.05% |
| ٨ (8) | 98.05% |
| ٩ (9) | 97.42% |

---

## 🏗️ Architecture

```
Input Layer:    784  neurons  (28×28 pixels flattened)
Hidden Layer 1: 512  neurons  + BatchNorm + ReLU + Dropout(0.3)
Hidden Layer 2: 256  neurons  + BatchNorm + ReLU + Dropout(0.3)
Hidden Layer 3: 128  neurons  + BatchNorm + ReLU + Dropout(0.15)
Output Layer:   10   neurons  → Softmax probabilities
```

**ليه الـ design ده؟**
- **3 hidden layers**: كل layer بتتعلم abstraction أعلى من اللي قبلها
- **BatchNorm**: بيثبت التدريب ويسرّعه
- **Dropout**: بيمنع overfitting بإجبار الشبكة تتعلم من أكتر من طريقة
- **Xavier init**: بيبدأ الـ weights بقيم مناسبة عشان التدريب يبدأ كويس

---

## 🛠️ Tech Stack

```
Python 3.12
PyTorch 2.12
torchvision
scikit-learn
matplotlib + seaborn
Streamlit
```

---

## 🚀 تشغيل المشروع

```bash
# 1. clone
git clone https://github.com/your-username/arabic-digit-classifier
cd arabic-digit-classifier

# 2. install
pip install -r requirements.txt

# 3. train
python train.py

# 4. run app
cd app
streamlit run app.py
```

---

## 📁 Project Structure

```
arabic-digit-classifier/
├── data/                    # MNIST data files
├── models/
│   └── best.pth             # أحسن موديل محفوظ
├── results/
│   ├── 01_samples.png       # عينات من الداتا
│   ├── 02_class_dist.png    # توزيع الـ classes
│   ├── 03_pixel_dist.png    # توزيع الـ pixels
│   ├── 04_history.png       # training curves
│   ├── 05_confusion.png     # confusion matrix
│   ├── 06_errors.png        # الصور اللي غلط فيها
│   ├── 07_confidence.png    # confidence analysis
│   └── results.json         # النتائج الرقمية
├── app/
│   └── app.py               # Streamlit app
├── train.py                 # Training script
├── requirements.txt
└── README.md
```

---

## 📈 Training Details

| Config | Value |
|--------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | CrossEntropy (label_smoothing=0.1) |
| Scheduler | ReduceLROnPlateau |
| Batch Size | 128 |
| Epochs | 15 |
| Regularization | Dropout + Weight Decay (1e-4) |
| Data Augmentation | RandomRotation ±15° + RandomAffine |

---

## 🧠 ماذا تعلمت

- بناء MLP من الصفر بـ PyTorch
- أهمية BatchNorm و Dropout في تحسين التدريب
- تحليل الـ Confusion Matrix لفهم أخطاء الموديل
- Confidence analysis لمعرفة متى نثق في الموديل
- Deploy نموذج AI كـ web app بـ Streamlit

---

*Built with ❤️ as Phase 1 of AI/ML Learning Roadmap*
