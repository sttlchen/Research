# ==== OASIS barcode -> CDRSUM regression with CNN, linear baselines, and optional TDA ====
# Run as a single script. Requires: numpy, pandas, scikit-learn, torch.
# Optional (for TDA): gudhi, persim (from scikit-tda). If missing, script skips TDA gracefully.

import os, re, math, random, warnings
import numpy as np
import pandas as pd

# --- Torch & Sklearn ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, ElasticNetCV

# ---- Optional TDA imports (handled safely) ----
TDA_AVAILABLE = True
try:
    import gudhi as gd
    from persim import PersistenceImager
except Exception as e:
    TDA_AVAILABLE = False
    warnings.warn(f"TDA dependencies not available ({e}). TDA-based features will be skipped.")

# =========================
# Config
# =========================
FOLDER = r"C:\Users\chent\PycharmProjects\Research\OASIS\barcodes_OASIS\G3\cs5_os3\CCs36_Cycles37\UNKNOWN"
CLIN_PATH = r"C:\Users\chent\PycharmProjects\Research\OASIS\final_clinical_data.csv"

EXPECTED_SHAPE = (73, 73)
USE_QUANTILE_Y = True       # transform targets to ~Normal for NN training, invert for metrics
N_SPLITS = 10                # GroupKFold splits (by OASISID)
SEED = 42

# Baseline (linear) model config
PCA_VARIANCE = 0.95         # keep 95% explained variance
RIDGE_ALPHAS = np.logspace(-3, 3, 13)

# CNN config
EPOCHS = 150
BATCH_SIZE = 32            # smaller helps minority gradient show up; scale up only with class-balanced batches
LR = 3e-4                  # slightly lower than 1e-3 for stability with weights/focal
WEIGHT_DECAY = 5e-4        # a touch stronger regularization is often helpful
PATIENCE = 15               # let PRC/F1 breathe

# TDA config
USE_TDA = True           # try TDA if libs are present; auto-disabled if libs missing
TDA_PI_BINS = (20, 20)      # persistence image resolution (birth x persistence)
TDA_PI_SPREAD = 0.2         # persistence image Gaussian spread
TDA_PI_SPREAD = 0.2         # persistence image Gaussian spread
TDA_PI_RANGE = ((0, 1.0), (0, 1.0))  # (birth min/max), (persistence min/max)
ELASTICNET_L1RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

# =========================
# Utilities
# =========================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# =========================
# Data loading & matching
# =========================
set_seed()

clin = pd.read_csv(CLIN_PATH).rename(columns={'OASIS_session_label': 'clinical_session'})
clin['clinical_session'] = pd.to_numeric(clin['clinical_session'], errors='coerce')
clin['CDRSUM'] = pd.to_numeric(clin['CDRSUM'], errors='coerce')
#clin = clin.dropna(subset=['OASISID', 'clinical_session', 'CDRSUM']).reset_index(drop=True)

files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith(".npy")])

pat = re.compile(
    r'^sub-(?P<id>OAS\d{5})_ses-(?P<sess>\d+)_task-rest(?:_run-(?P<run>\d+))?_barcode\.npy$',
    re.IGNORECASE
)

rows = []
for f in files:
    m = pat.match(f)
    if not m:
        continue
    rows.append({
        'file': f,
        'OASISID': m.group('id').upper(),
        'imaging_session': int(m.group('sess')),
    })
img_df = pd.DataFrame(rows)

def load_npy_array(row):
    filename = row["file"]
    m = re.match(r'^sub-(OAS\d{5})_ses-(\d+)_', filename, re.IGNORECASE)
    if not m:
        print(f"⚠️ Skipping {filename}: unexpected naming pattern.")
        return None
    file_id, file_sess = m.groups()
    file_sess = int(file_sess)
    if file_id.upper() != row["OASISID"] or file_sess != row["imaging_session"]:
        print(f"❌ ID/session mismatch: {filename}")
        return None
    arr = np.load(os.path.join(FOLDER, filename))
    arr = np.squeeze(arr)
    if arr.shape != EXPECTED_SHAPE:
        print(f"⚠️ Shape warning: {filename} has {arr.shape}, expected {EXPECTED_SHAPE}")
    return arr.astype(np.float32, copy=False)

img_df["img_array"] = img_df.apply(load_npy_array, axis=1)
print("null arrays:", img_df['img_array'].isna().sum())
img_df = img_df.dropna(subset=['img_array']).reset_index(drop=True)

# Match imaging to clinical by nearest session number (per file)
cross = img_df.merge(clin, on='OASISID', how='inner')
cross['abs_diff'] = (cross['clinical_session'] - cross['imaging_session']).abs()
min_diff = cross.groupby('file')['abs_diff'].transform('min')
matched = cross[cross['abs_diff'] == min_diff].reset_index(drop=True)


print("\nVisit gap summary (in sessions):")
print(matched['abs_diff'].describe())

# =========================
# Arrays -> X, targets -> y
# =========================
X = np.stack([np.asarray(a, dtype=np.float32) for a in matched['img_array']])  # (N, 73, 73)
y_raw = matched['CDRSUM'].astype(np.float32).values
groups = matched['OASISID'].values

# # Per-image z-scoring improves stability
# X_norm = []
# for img in X_raw:
#     m, s = img.mean(), img.std()
#     if not np.isfinite(s) or s < 1e-6: s = 1.0
#     X_norm.append((img - m) / s)
# X = np.stack(X_norm).astype(np.float32)  # (N, 73, 73)

# Optional target transform for NN; linear models stay on original scale
if USE_QUANTILE_Y:
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(4000, len(y_raw)))
    y_trans = qt.fit_transform(y_raw.reshape(-1, 1)).astype(np.float32).ravel()
else:
    y_trans = y_raw

# =========================
# Torch dataset & CNN model
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImgRegDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # (N,1,73,73)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # (N,1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class CDRNet(nn.Module):
    # A slightly deeper, regularized CNN with global-average-pooling to reduce params
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2), nn.Dropout(0.10)
        )  # 73->36
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(0.15)
        )  # 36->18
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout(0.20)
        )  # 18->9
        self.gap = nn.AdaptiveAvgPool2d(1)  # (N,128,1,1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.gap(x)
        return self.head(x)

def train_one_fold(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    huber = nn.HuberLoss(delta=1.0)

    best_val, best_state, wait = math.inf, None, 0
    for ep in range(1, epochs+1):
        model.train()
        tr_loss_sum, n_tr = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = huber(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss_sum += loss.item() * xb.size(0); n_tr += xb.size(0)
        tr_loss = tr_loss_sum / max(1, n_tr)

        # Validate (MSE on transformed target)
        model.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vloss = F.mse_loss(pred, yb)
                val_loss_sum += vloss.item() * xb.size(0); n_val += xb.size(0)
        val_loss = val_loss_sum / max(1, n_val)
        sched.step(val_loss)
        print(f"  Epoch {ep:02d}: train(Huber)={tr_loss:.4f}  val(MSE)={val_loss:.4f}")

        # Early stopping
        if val_loss + 1e-6 < best_val:
            best_val = val_loss; best_state = {k: v.cpu() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

print("\n=== CNN (grouped CV on transformed y; metrics inverted to original scale) ===")
gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
fold_preds, fold_true = [], []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_trans, groups=groups), 1):
    print(f"\n--- Fold {fold} ---")
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y_trans[tr_idx], y_trans[va_idx]

    ds_tr = ImgRegDataset(Xtr, ytr)
    ds_va = ImgRegDataset(Xva, yva)

    train_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ds_va, batch_size=max(64, BATCH_SIZE), shuffle=False)

    model = CDRNet().to(device)
    model = train_one_fold(model, train_loader, val_loader)

    # Predict on validation fold
    model.eval()
    preds_va = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds_va.append(model(xb).cpu().numpy().ravel())
    preds_va = np.concatenate(preds_va)

    # Invert target transform for metrics
    if USE_QUANTILE_Y:
        preds_va_real = qt.inverse_transform(preds_va.reshape(-1,1)).ravel()
        yva_real      = qt.inverse_transform(yva.reshape(-1,1)).ravel()
    else:
        preds_va_real = preds_va; yva_real = yva

    fold_preds.append(preds_va_real); fold_true.append(yva_real)

y_true_cnn = np.concatenate(fold_true)
y_pred_cnn = np.concatenate(fold_preds)
print(f"\nCNN CV metrics:")
print(f"MAE  : {mean_absolute_error(y_true_cnn, y_pred_cnn):.3f}")
print(f"RMSE : {rmse(y_true_cnn, y_pred_cnn):.3f}")
print(f"R^2  : {r2_score(y_true_cnn, y_pred_cnn):.3f}")

# print("\n=== TDA (cubical complex -> persistence images -> ElasticNet), grouped CV ===")
# if USE_TDA and TDA_AVAILABLE:
#     tda_feats = X.reshape(len(X), -1).astype(np.float32)
#     # ---- Grouped CV with ElasticNet (unchanged) ----
#     gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
#     en_cv_preds, en_cv_true = [], []
#     for fold, (tr, va) in enumerate(gkf.split(tda_feats, y_raw, groups=groups), 1):
#         Xtr, Xva = tda_feats[tr], tda_feats[va]
#         ytr, yva = y_raw[tr], y_raw[va]
#
#         # 1) prune dead pixels
#         from sklearn.feature_selection import VarianceThreshold
#
#         vt = VarianceThreshold(1e-8)
#         Xtr = vt.fit_transform(Xtr);
#         Xva = vt.transform(Xva)
#
#         # 2) scale (+ optional PCA)
#         scaler = StandardScaler()
#         Xtr_s = scaler.fit_transform(Xtr);
#         Xva_s = scaler.transform(Xva)
#
#         # Optional: decorrelate
#         from sklearn.decomposition import PCA
#         pca = PCA(n_components=0.99, svd_solver="full", random_state=0)
#         Xtr_s = pca.fit_transform(Xtr_s); Xva_s = pca.transform(Xva_s)
#
#         # 3) robust & parallel ElasticNetCV
#         en = ElasticNetCV(
#             l1_ratio=[0.2, 0.5, 0.8],
#             alphas=np.logspace(-2, 2, 30),
#             cv=3,
#             max_iter=50000,
#             n_jobs=-1,
#             random_state=0
#         )
#         en.fit(Xtr_s, ytr)
#         pred_va = en.predict(Xva_s)
#
#         en_cv_preds.append(pred_va); en_cv_true.append(yva)
#
#     y_true_tda = np.concatenate(en_cv_true)
#     y_pred_tda = np.concatenate(en_cv_preds)
#     print(f"TDA (ElasticNet) CV metrics:")
#     print(f"MAE  : {mean_absolute_error(y_true_tda, y_pred_tda):.3f}")
#     print(f"RMSE : {rmse(y_true_tda, y_pred_tda):.3f}")
#     print(f"R^2  : {r2_score(y_true_tda, y_pred_tda):.3f}")
#
#     # Optional blend with Ridge if you computed it earlier
#     if 'y_pred_lin' in globals() and len(y_pred_lin) == len(y_pred_tda):
#         blend = 0.5 * y_pred_lin + 0.5 * y_pred_tda
#         print(f"\nBlend (Ridge ⨉ TDA) CV metrics:")
#         print(f"MAE  : {mean_absolute_error(y_true_lin, blend):.3f}")
#         print(f"RMSE : {rmse(y_true_lin, blend):.3f}")
#         print(f"R^2  : {r2_score(y_true_lin, blend):.3f}")
# else:
#     print("TDA disabled (missing dependencies or USE_TDA=False). Skipping TDA block.")