import re
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


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
X = np.stack([a for a in matched['img_array']])     # shape (N, 73, 73)
y = matched['CDRSUM'].values.astype(np.float32)     # shape (N,)

# add channel dimension: (N, 1, 73, 73)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


class CDRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 18 * 18, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (N,16,36,36)
        x = self.pool(F.relu(self.conv2(x)))   # (N,32,18,18)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = CDRNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(50):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    print(f"Epoch {epoch+1}, MSE={total_loss/len(dataset):.4f}")
model.eval()
with torch.no_grad():
    preds = model(X).squeeze().numpy()