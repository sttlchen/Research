import re
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer
import math
import random


folder = r"C:\Users\chent\PycharmProjects\Research\OASIS\barcodes_OASIS\G3\cs5_os3\CCs36_Cycles37\UNKNOWN"
clin_path = r"C:\Users\chent\PycharmProjects\Research\OASIS\final_clinical_data.csv"

# Clinical
clin = pd.read_csv(clin_path).rename(columns={'OASIS_session_label': 'clinical_session'})#csv file read
clin['clinical_session'] = pd.to_numeric(clin['clinical_session'], errors='coerce')#convert session number to numeric
clin['CDRSUM'] = pd.to_numeric(clin['CDRSUM'], errors='coerce')#convert CDRSUM to numeric
clin = clin.dropna(subset=['OASISID', 'clinical_session', 'CDRSUM']).reset_index(drop=True)#drop missing data

# Files
files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".npy")])

# Regex for your exact pattern
pat = re.compile(
    r'^sub-(?P<id>OAS\d{5})_ses-(?P<sess>\d+)_task-rest_run-(?P<run>\d+)_barcode\.npy$',
    re.IGNORECASE
)#extract numbers from names

rows = []
for f in files:
    m = pat.match(f)
    if not m:
        continue
    rows.append({
        'file': f,
        'OASISID': m.group('id').upper(),
        'imaging_session': int(m.group('sess')),   # handles leading zeros (e.g., 0129 -> 129)
        'run': int(m.group('run'))
    })

img_df = pd.DataFrame(rows)
pd.set_option('display.max_columns', None)
#print(img_df)

def load_npy_array(row):
    """
    Load a .npy file and verify OASISID and session number match expected metadata.
    """
    filename = row["file"]
    expected_id = row["OASISID"]
    expected_sess = row["imaging_session"]

    # Parse ID and session back from the filename to cross-check
    m = re.match(r'^sub-(OAS\d{5})_ses-(\d+)_', filename)
    if not m:
        print(f"⚠️ Skipping {filename}: unexpected naming pattern.")
        return None

    file_id, file_sess = m.groups()
    file_sess = int(file_sess)

    # ID/session consistency checks
    if file_id.upper() != expected_id:
        print(f"❌ ID mismatch: {filename} expected {expected_id}, found {file_id}")
        return None
    if file_sess != expected_sess:
        print(f"❌ Session mismatch: {filename} expected {expected_sess}, found {file_sess}")
        return None

    # Load the array
    path = os.path.join(folder, filename)
    arr = np.load(path)
    arr = np.squeeze(arr)

    # Shape check
    if arr.shape != (73, 73):
        print(f"⚠️ Shape warning: {filename} has {arr.shape}, expected (73,73)")

    return arr


img_df["img_array"] = img_df.apply(load_npy_array, axis=1)
#Below checks if correctly merged
#print(img_df[['file', 'OASISID', 'imaging_session', 'run', 'img_array']].head(10))

#Below are for checking if images are loaded correctly by drawing one out.
# plt.imshow(img_df.loc[1, "img_array"], cmap="gray")
# plt.title(img_df.loc[1, "file"])
# plt.show()

# Next, try associate those images with csv file CDRSUM
# Match imaging and clinical data by participant and nearest session number
cross = img_df.merge(clin, on='OASISID', how='inner')
cross['abs_diff'] = (cross['clinical_session'] - cross['imaging_session']).abs()

# For each image, pick the clinical record with the smallest session difference
idx = cross.groupby('file')['abs_diff'].idxmin()
matched = cross.loc[idx].reset_index(drop=True)

# Show result
# any failed loads?
print("null arrays:", img_df['img_array'].isna().sum())

# drop rows where loading/validation failed
img_df = img_df.dropna(subset=['img_array']).reset_index(drop=True)

# after matching: how close are visits?
print(matched['abs_diff'].describe())
print("max gap (days):", matched['abs_diff'].max())


# Playgrounds for experimental analysis
# Here we perform some basic regression first to see if any patterns
# Flatten each 73×73 array into a 1D vector (5329 features)
# X = np.stack([arr.ravel() for arr in matched['img_array']])
# y = matched['CDRSUM'].astype(float).values
# groups = matched['OASISID'].values  # group by participant for safe CV
# print("Feature matrix:", X.shape)
# print("Target shape:", y.shape)
#
# cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
# pipe = Pipeline([
#     ('scaler', StandardScaler()),  # normalize pixel intensities
#     ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))
# ])
# y_pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups)
#
# print(f"Results (GroupKFold CV):")
# mae = mean_absolute_error(y, y_pred)
# mse = mean_squared_error(y, y_pred)   # always supported
# rmse = np.sqrt(mse)                   # version-agnostic RMSE
# r2 = r2_score(y, y_pred)
# print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}") Not giving desired result.

#Another way proposed with PyTorch use
# X = np.stack([a for a in matched['img_array']])     # shape (N, 73, 73)
# y = matched['CDRSUM'].values.astype(np.float32)     # shape (N,)
#
# # add channel dimension: (N, 1, 73, 73)
# X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
# y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
#
#
# class CDRNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32 * 18 * 18, 64)
#         self.fc2 = nn.Linear(64, 1)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))   # (N,16,36,36)
#         x = self.pool(F.relu(self.conv2(x)))   # (N,32,18,18)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
# dataset = TensorDataset(X, y)
# loader = DataLoader(dataset, batch_size=16, shuffle=True)
#
# model = CDRNet()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.MSELoss()
#
# for epoch in range(50):
#     model.train()
#     total_loss = 0
#     for xb, yb in loader:
#         pred = model(xb)
#         loss = loss_fn(pred, yb)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * len(xb)
#     print(f"Epoch {epoch+1}, MSE={total_loss/len(dataset):.4f}")
# model.eval()
# with torch.no_grad():
#     preds = model(X).squeeze().numpy()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Build X, y, groups ---
X = np.stack([np.asarray(a, dtype=np.float32) for a in matched['img_array']])  # (N, 73, 73)
y = matched['CDRSUM'].astype(np.float32).values                                # (N,)
groups = matched['OASISID'].values

# Optional: target transform to reduce skew (helps with imbalance)
use_quantile = True
if use_quantile:
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(y)))
    y_trans = qt.fit_transform(y.reshape(-1,1)).astype(np.float32).ravel()
else:
    y_trans = y

# --- simple min-max/standardize per-image (optional but helpful) ---
# normalize to mean 0, std 1 across each image (avoid division by 0)
X_norm = []
for img in X:
    m, s = img.mean(), img.std()
    if s < 1e-6: s = 1.0
    X_norm.append((img - m) / s)
X = np.stack(X_norm)  # (N,73,73)

# --- Torch dataset ---
class ImgRegDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,73,73)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N,1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# --- Simple CNN ---
class CDRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.drop  = nn.Dropout(p=0.25)
        # 73-> pool->36 -> pool->18 -> pool->9 (three pools)
        self.fc1 = nn.Linear(64*9*9, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))      # (N,16,36,36)
        x = self.pool(F.relu(self.conv2(x)))      # (N,32,18,18)
        x = self.pool(F.relu(self.conv3(x)))      # (N,64,9,9)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# --- Training utils ---
def train_one_fold(model, train_loader, val_loader, epochs=30, lr=1e-3, weighted=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    huber = nn.HuberLoss(delta=1.0, reduction='none')  # robust to outliers
    best_val = math.inf
    wait, patience = 0, 5
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        tr_loss_sum, n_tr = 0.0, 0
        for xb, yb, *wb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss_vec = huber(pred, yb)  # (B,1)
            if weighted and len(wb)>0 and wb[0] is not None:
                w = wb[0].to(device).view_as(loss_vec)
                loss = (loss_vec * w).mean()
            else:
                loss = loss_vec.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss_sum += loss.item() * xb.size(0); n_tr += xb.size(0)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss_sum, n_val = 0.0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vloss = nn.MSELoss()(pred, yb)  # track MSE on transformed target
                val_loss_sum += vloss.item() * xb.size(0); n_val += xb.size(0)
        tr_mse = tr_loss_sum / max(n_tr,1)
        val_mse = val_loss_sum / max(n_val,1)
        print(f"Epoch {ep:02d}: train(Huber)={tr_mse:.4f}  val(MSE)={val_mse:.4f}")

        # early stopping on val MSE
        if val_mse < best_val - 1e-5:
            best_val = val_mse; best_state = {k:v.cpu() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience: break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# --- GroupKFold split (no subject leakage) ---
gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
fold_preds, fold_true = [], []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_trans, groups=groups), 1):
    print(f"\n=== Fold {fold} ===")
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y_trans[tr_idx], y_trans[va_idx]

    # Weighted sampling to mitigate imbalance on the transformed target (bin by quantiles)
    # Build per-sample weights inversely to bin frequency
    bins = np.clip(np.digitize(ytr, np.quantile(ytr, [0, .2, .4, .6, .8, 1.0])[1:-1]), 0, 3)
    counts = np.bincount(bins, minlength=4) + 1e-6
    w_tr = np.array([1.0 / counts[b] for b in bins], dtype=np.float32)
    w_tr = w_tr / w_tr.mean()

    ds_tr = ImgRegDataset(Xtr, ytr)
    ds_va = ImgRegDataset(Xva, yva)

    # custom loader that yields weights
    class WeightedDS(Dataset):
        def __init__(self, base_ds, weights):
            self.base_ds = base_ds
            self.weights = torch.tensor(weights, dtype=torch.float32)
        def __len__(self): return len(self.base_ds)
        def __getitem__(self, i):
            x, y = self.base_ds[i]
            return x, y, self.weights[i:i+1]  # keep shape (1,)

    train_loader = DataLoader(WeightedDS(ds_tr, w_tr), batch_size=32, shuffle=True)
    val_loader   = DataLoader(ds_va, batch_size=64, shuffle=False)

    model = CDRNet().to(device)
    model = train_one_fold(model, train_loader, val_loader, epochs=40, lr=1e-3, weighted=True)

    # predictions on validation fold
    model.eval()
    with torch.no_grad():
        preds_va = []
        for xb, yb in val_loader:
            xb = xb.to(device)
            pv = model(xb).cpu().numpy().ravel()
            preds_va.append(pv)
        preds_va = np.concatenate(preds_va)

    # invert target transform if used
    if use_quantile:
        preds_va_real = qt.inverse_transform(preds_va.reshape(-1,1)).ravel()
        yva_real      = qt.inverse_transform(yva.reshape(-1,1)).ravel()
    else:
        preds_va_real = preds_va; yva_real = yva

    fold_preds.append(preds_va_real)
    fold_true.append(yva_real)

# --- aggregate CV metrics ---
y_true_all = np.concatenate(fold_true)
y_pred_all = np.concatenate(fold_preds)

mae = mean_absolute_error(y_true_all, y_pred_all)
mse = mean_squared_error(y_true_all, y_pred_all)
rmse = np.sqrt(mse)  # robust to sklearn version
r2 = r2_score(y_true_all, y_pred_all)

print("\n=== Cross-validated performance (subject-level splits) ===")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R^2  : {r2:.3f}")