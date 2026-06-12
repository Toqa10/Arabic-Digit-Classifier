"""
================================================
🔢 Arabic Handwritten Digits Classifier
================================================
المشروع: تصنيف الأرقام العربية المكتوبة بإيد
نستخدم MNIST كـ base لأن structure الأرقام متشابه
الهدف:  MLP قوي مع شرح كامل لكل خطوة
================================================
"""

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns, os, time, json

# ── Arabic digit symbols & names ──────────────────────────
ARABIC   = ['٠','١','٢','٣','٤','٥','٦','٧','٨','٩']
AR_NAMES = ['صفر','واحد','اتنين','تلاتة','أربعة','خمسة','ستة','سبعة','تمانية','تسعة']

# ── Config ─────────────────────────────────────────────────
CFG = dict(batch=128, epochs=15, lr=1e-3,
           h1=512, h2=256, h3=128, drop=0.3,
           val_frac=0.1, save='models/best.pth')
os.makedirs('models',  exist_ok=True)
os.makedirs('results', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}\n")

# ══════════════════════════════════════════════════════════
# 1. تحميل وتحضير الداتا
# ══════════════════════════════════════════════════════════
print("📥 بنحمل الداتا...")
X_tr = np.load('data/X_train.npy').astype('float32') / 255.0   # normalize 0-1
y_tr = np.load('data/y_train.npy').astype('int64')
X_te = np.load('data/X_test.npy' ).astype('float32') / 255.0
y_te = np.load('data/y_test.npy' ).astype('int64')

# Standardize: mean=0.1307, std=0.3081 (قيم MNIST المعروفة)
MEAN, STD = 0.1307, 0.3081
X_tr = (X_tr - MEAN) / STD
X_te = (X_te - MEAN) / STD

# Flatten: (N,28,28) → (N,784)
X_tr_flat = X_tr.reshape(-1, 784)
X_te_flat = X_te.reshape(-1, 784)

# → Tensors
X_tr_t = torch.tensor(X_tr_flat)
y_tr_t = torch.tensor(y_tr)
X_te_t = torch.tensor(X_te_flat)
y_te_t = torch.tensor(y_te)

full_ds = TensorDataset(X_tr_t, y_tr_t)
val_n   = int(len(full_ds) * CFG['val_frac'])
tr_n    = len(full_ds) - val_n
train_ds, val_ds = random_split(full_ds, [tr_n, val_n],
                                generator=torch.Generator().manual_seed(42))
test_ds  = TensorDataset(X_te_t, y_te_t)

tr_ld  = DataLoader(train_ds, batch_size=CFG['batch'], shuffle=True)
val_ld = DataLoader(val_ds,   batch_size=CFG['batch'], shuffle=False)
te_ld  = DataLoader(test_ds,  batch_size=CFG['batch'], shuffle=False)

print(f"✅ Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")

# ══════════════════════════════════════════════════════════
# 2. EDA — استكشاف الداتا
# ══════════════════════════════════════════════════════════
def eda():
    print("\n📊 EDA...")
    # --- sample images ---
    fig, axes = plt.subplots(3, 10, figsize=(22, 7))
    fig.suptitle('عينات من كل رقم — raw | normalized | heatmap',
                 fontsize=13, fontweight='bold')
    seen = {}
    for img, lbl in zip(X_tr, y_tr):
        if lbl not in seen: seen[lbl] = img
        if len(seen) == 10: break

    for d in range(10):
        raw  = np.load('data/X_train.npy')[np.where(y_tr == d)[0][0]].astype('float32') / 255.0
        norm = seen[d]
        axes[0,d].imshow(raw,  cmap='gray');  axes[0,d].set_title(f'{d} {ARABIC[d]}', fontsize=11); axes[0,d].axis('off')
        axes[1,d].imshow(norm, cmap='gray');  axes[1,d].set_title(AR_NAMES[d], fontsize=9);          axes[1,d].axis('off')
        axes[2,d].imshow(norm, cmap='hot');   axes[2,d].set_title('heatmap', fontsize=8);            axes[2,d].axis('off')

    plt.tight_layout()
    plt.savefig('results/01_samples.png', dpi=140, bbox_inches='tight')
    plt.close()

    # --- class distribution ---
    counts = np.bincount(y_tr)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0,1,10))
    bars   = ax.bar([f'{ARABIC[i]}\n({i})' for i in range(10)], counts, color=colors, edgecolor='white', lw=1.5)
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+40, f'{c:,}',
                ha='center', fontsize=10, fontweight='bold')
    ax.set_title('توزيع الـ Classes', fontsize=13, fontweight='bold')
    ax.set_ylabel('عدد الصور'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/02_class_dist.png', dpi=140, bbox_inches='tight')
    plt.close()

    # --- pixel intensity distribution ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(X_tr.flatten(), bins=60, color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_title('توزيع شدة الـ Pixels بعد الـ Normalization', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pixel value'); ax.set_ylabel('Count'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/03_pixel_dist.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("   ✅ حُفظت الـ EDA charts")

eda()

# ══════════════════════════════════════════════════════════
# 3. الموديل
# ══════════════════════════════════════════════════════════
class ArabicMLP(nn.Module):
    """
    3-hidden-layer MLP مع BatchNorm + Dropout + Xavier init
    ليه 3 layers؟
      L1 (512): يتعلم features بسيطة — خطوط وزوايا
      L2 (256): يجمع features في patterns
      L3 (128): يجمع patterns في تمثيل نهائي
      Out (10): يقرر الرقم
    """
    def __init__(self):
        super().__init__()
        d = CFG['drop']
        self.net = nn.Sequential(
            nn.Linear(784, CFG['h1']),
            nn.BatchNorm1d(CFG['h1']),   # normalize → تدريب أستقر
            nn.ReLU(),
            nn.Dropout(d),               # يطفي 30% neurons عشوائي → يقاوم overfitting

            nn.Linear(CFG['h1'], CFG['h2']),
            nn.BatchNorm1d(CFG['h2']),
            nn.ReLU(),
            nn.Dropout(d),

            nn.Linear(CFG['h2'], CFG['h3']),
            nn.BatchNorm1d(CFG['h3']),
            nn.ReLU(),
            nn.Dropout(d/2),             # أقل dropout قرب الـ output

            nn.Linear(CFG['h3'], 10),   # output — مفيش softmax (CrossEntropy بتضيفها)
        )
        self._init()

    def _init(self):                     # Xavier initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

model = ArabicMLP().to(device)
print(f"\n🏗️  Model: {model.n_params():,} parameters")

# ══════════════════════════════════════════════════════════
# 4. Loss, Optimizer, Scheduler
# ══════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# label_smoothing: بيخلي الـ targets ناعمة شوية → يمنع overconfidence

optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
# weight_decay: L2 regularization → بيعاقب الـ weights الكبيرة

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3)
# لو val_loss وقفت تتحسن 3 epochs → نقلل الـ LR لنص

# ══════════════════════════════════════════════════════════
# 5. Training & Evaluation functions
# ══════════════════════════════════════════════════════════
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss = correct = total = 0
    preds_all = []; labels_all = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = criterion(out, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            pred      = out.argmax(1)
            correct  += (pred == yb).sum().item()
            total    += yb.size(0)
            tot_loss += loss.item() * yb.size(0)
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(yb.cpu().numpy())

    return tot_loss/total, 100*correct/total, preds_all, labels_all

# ══════════════════════════════════════════════════════════
# 6. Training Loop
# ══════════════════════════════════════════════════════════
print(f"\n🚀 بدأ التدريب ({CFG['epochs']} epochs)...\n")
print(f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'VaLoss':>7} | {'VaAcc':>6} | {'LR':>8} | Time")
print("─"*62)

hist = {k:[] for k in ['tl','ta','vl','va','lr']}
best_acc = 0; best_ep = 0

for ep in range(1, CFG['epochs']+1):
    t0 = time.time()
    tl, ta, _, _    = run_epoch(tr_ld,  train=True)
    vl, va, _, _    = run_epoch(val_ld, train=False)
    scheduler.step(vl)
    lr = optimizer.param_groups[0]['lr']
    elapsed = time.time() - t0

    for k,v in zip(['tl','ta','vl','va','lr'],[tl,ta,vl,va,lr]):
        hist[k].append(v)

    star = ''
    if va > best_acc:
        best_acc, best_ep = va, ep
        torch.save({'ep':ep,'state':model.state_dict(),'acc':va,'cfg':CFG}, CFG['save'])
        star = ' 💾'

    print(f"{ep:>3} | {tl:>7.4f} | {ta:>5.2f}% | {vl:>7.4f} | {va:>5.2f}% | {lr:.2e} | {elapsed:.1f}s{star}")

print(f"\n🏆 Best Val Acc: {best_acc:.2f}% @ epoch {best_ep}")

# ══════════════════════════════════════════════════════════
# 7. رسم Training History
# ══════════════════════════════════════════════════════════
def plot_history():
    eps = range(1, len(hist['tl'])+1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training History — Arabic Digit Classifier', fontsize=13, fontweight='bold')

    axes[0].plot(eps, hist['tl'], 'b-o', ms=4, label='Train')
    axes[0].plot(eps, hist['vl'], 'r-o', ms=4, label='Val')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(eps, hist['ta'], 'b-o', ms=4, label='Train')
    axes[1].plot(eps, hist['va'], 'r-o', ms=4, label='Val')
    axes[1].set_title('Accuracy %'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=.3)

    axes[2].plot(eps, hist['lr'], 'g-o', ms=4)
    axes[2].set_title('Learning Rate'); axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log'); axes[2].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig('results/04_history.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("💾 results/04_history.png")

plot_history()

# ══════════════════════════════════════════════════════════
# 8. Final Evaluation على Test Set
# ══════════════════════════════════════════════════════════
print("\n📊 Final Evaluation على Test Set...")
ckpt = torch.load(CFG['save'], map_location=device)
model.load_state_dict(ckpt['state'])

te_loss, te_acc, te_preds, te_labels = run_epoch(te_ld, train=False)
print(f"🎯 Test Accuracy: {te_acc:.2f}%")
print(f"📉 Test Loss:     {te_loss:.4f}\n")

print(classification_report(
    te_labels, te_preds,
    target_names=[f'{ARABIC[i]}({i})' for i in range(10)]
))

# ══════════════════════════════════════════════════════════
# 9. Confusion Matrix
# ══════════════════════════════════════════════════════════
def plot_cm(labels, preds):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ticks   = [f'{ARABIC[i]}\n({i})' for i in range(10)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm,      annot=True, fmt='d',   cmap='Blues',  ax=axes[0], xticklabels=ticks, yticklabels=ticks)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1], xticklabels=ticks, yticklabels=ticks, vmin=0, vmax=1)
    for ax, t in zip(axes, ['عدد الصور', 'نسبة مئوية']):
        ax.set_title(f'Confusion Matrix — {t}', fontsize=12, fontweight='bold')
        ax.set_ylabel('الحقيقي'); ax.set_xlabel('المتوقع')
    plt.tight_layout()
    plt.savefig('results/05_confusion.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("💾 results/05_confusion.png")

plot_cm(te_labels, te_preds)

# ══════════════════════════════════════════════════════════
# 10. الصور اللي غلط فيها الموديل
# ══════════════════════════════════════════════════════════
def plot_errors():
    model.eval()
    wrongs = []
    with torch.no_grad():
        for xb, yb in te_ld:
            xb, yb = xb.to(device), yb.to(device)
            out   = model(xb)
            probs = torch.softmax(out, dim=1)
            pred  = out.argmax(1)
            mask  = pred != yb
            for i in range(len(yb)):
                if mask[i] and len(wrongs) < 25:
                    wrongs.append((xb[i].cpu(), pred[i].item(),
                                   yb[i].item(), probs[i].max().item()))
            if len(wrongs) >= 25: break

    fig, axes = plt.subplots(5, 5, figsize=(15, 16))
    fig.suptitle('الصور اللي غلط فيها الموديل', fontsize=13, fontweight='bold')
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(wrongs):
            img, pr, tr, cf = wrongs[idx]
            img = (img.numpy().reshape(28,28) * STD + MEAN)
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap='gray')
            ax.set_title(
                f'✅ {ARABIC[tr]}({tr})  ❌ {ARABIC[pr]}({pr})\n{cf*100:.0f}% confidence',
                color='darkred', fontsize=9, fontweight='bold'
            )
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('results/06_errors.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("💾 results/06_errors.png")

plot_errors()

# ══════════════════════════════════════════════════════════
# 11. Confidence Analysis
# ══════════════════════════════════════════════════════════
def plot_confidence():
    model.eval()
    c_conf, w_conf = [], []
    per_class_acc  = {i: {'c':0,'t':0} for i in range(10)}

    with torch.no_grad():
        for xb, yb in te_ld:
            xb, yb = xb.to(device), yb.to(device)
            out   = model(xb)
            probs = torch.softmax(out, dim=1)
            pred  = out.argmax(1)
            conf  = probs.max(1).values
            for i in range(len(yb)):
                lbl = yb[i].item()
                per_class_acc[lbl]['t'] += 1
                if pred[i] == yb[i]:
                    c_conf.append(conf[i].item())
                    per_class_acc[lbl]['c'] += 1
                else:
                    w_conf.append(conf[i].item())

    # per-class accuracy
    class_accs = [per_class_acc[i]['c']/per_class_acc[i]['t']*100 for i in range(10)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('تحليل الـ Confidence', fontsize=13, fontweight='bold')

    axes[0].hist(c_conf, bins=40, color='green', alpha=.7, label=f'Correct ({len(c_conf):,})')
    axes[0].hist(w_conf, bins=40, color='red',   alpha=.7, label=f'Wrong ({len(w_conf):,})')
    axes[0].set_title('توزيع الـ Confidence'); axes[0].legend(); axes[0].grid(alpha=.3)

    thr = np.arange(0.5, 1.0, 0.05)
    prec = []
    for t in thr:
        c = sum(1 for x in c_conf if x >= t)
        w = sum(1 for x in w_conf if x >= t)
        prec.append(c/(c+w)*100 if c+w else 100)
    axes[1].plot(thr*100, prec, 'b-o', ms=5)
    axes[1].axhline(99, color='red', ls='--', alpha=.5)
    axes[1].set_title('Precision vs Threshold'); axes[1].grid(alpha=.3)

    colors = ['green' if a >= 98 else 'orange' if a >= 95 else 'red' for a in class_accs]
    axes[2].bar([f'{ARABIC[i]}\n({i})' for i in range(10)], class_accs, color=colors, edgecolor='white')
    axes[2].set_title('Accuracy لكل رقم'); axes[2].set_ylim(90, 100); axes[2].grid(axis='y', alpha=.3)
    for i, v in enumerate(class_accs):
        axes[2].text(i, v+0.1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/07_confidence.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("💾 results/07_confidence.png")
    print(f"\n📊 Avg Correct confidence: {np.mean(c_conf)*100:.1f}%")
    print(f"📊 Avg Wrong confidence:   {np.mean(w_conf)*100:.1f}%")
    print("\n📊 Per-class accuracy:")
    for i in range(10):
        bar = '█' * int(class_accs[i]-90)
        print(f"   {ARABIC[i]} ({i}): {class_accs[i]:5.2f}% {bar}")

plot_confidence()

# ══════════════════════════════════════════════════════════
# 12. حفظ النتائج
# ══════════════════════════════════════════════════════════
res = dict(test_accuracy=round(te_acc,4), test_loss=round(te_loss,4),
           best_val=round(best_acc,4), best_epoch=best_ep,
           params=model.n_params(), config=CFG)
with open('results/results.json','w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print("\n" + "═"*55)
print("🎉 اتمت بنجاح!")
print("═"*55)
print(f"🎯 Test Accuracy:  {te_acc:.2f}%")
print(f"📉 Test Loss:      {te_loss:.4f}")
print(f"🏆 Best Val Acc:   {best_acc:.2f}%")
print(f"📦 Parameters:     {model.n_params():,}")
print(f"\n📁 results/ فيها 7 charts")
print(f"📁 models/best.pth الموديل المحفوظ")
