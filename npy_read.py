import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

folder = "/Users/steventianlechen/Desktop/MSS Fall 2025/Research/barcodes_OASIS/G1/cs5_os3/CCs19_Cycles19/UNKNOWN/"
final_clinical_data = pd.read_csv("/Users/steventianlechen/Desktop/MSS Fall 2025/Research/final_clinical_data.csv")
images = []

# Files
files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".npy")])

# Regex for your exact pattern
pat = re.compile(
    r'^sub-(?P<id>OAS\d{5})_ses-(?P<sess>\d+)_task-rest_run-(?P<run>\d+)_barcode\.npy$',
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
        'imaging_session': int(m.group('sess')),   # handles leading zeros (e.g., 0129 -> 129)
        'run': int(m.group('run'))
    })

img_df = pd.DataFrame(rows)
print(img_df)

def load_arr(path):
    arr = np.load(os.path.join(folder, path))
    if arr.ndim == 2:
        return arr
    return arr[..., 0]  # fallback if extra dims exist

# Load & average runs per (ID, session)
grouped = []
for (pid, sess), g in img_df.groupby(['OASISID', 'imaging_session'], sort=False):
    arrays = []
    for f in g['file']:
        arr = load_arr(f)
        if arr.shape != (37, 37):
            continue  # skip unexpected shapes; or resize if you prefer
        arrays.append(arr)
    if arrays:
        avg = np.mean(np.stack(arrays, axis=0), axis=0)
        grouped.append({'OASISID': pid, 'imaging_session': sess, 'img_array': avg})

img_per_session = pd.DataFrame(grouped)
print("Sessions with images:", len(img_per_session))

def nearest_match(group_img, group_clin):
    cross = group_img.assign(key=1).merge(group_clin.assign(key=1), on='key').drop('key', axis=1)
    cross['abs_diff'] = (cross['clinical_session'] - cross['imaging_session']).abs()
    # pick the closest clinical visit for each image row
    keep_idx = cross.groupby(['OASISID','imaging_session'], as_index=False)['abs_diff'].idxmin()['abs_diff'].values
    return cross.loc[keep_idx]

matched = (
    img_per_session.groupby('OASISID', group_keys=False)
                   .apply(lambda g: nearest_match(g, clin[clin['OASISID']==g.name]))
                   .reset_index(drop=True)
)

# Optional tolerance (e.g., only keep pairs within 180 days)
# matched = matched[matched['abs_diff'] <= 180].reset_index(drop=True)

print(matched[['OASISID','imaging_session','clinical_session','abs_diff']].head())

# Flatten 37x37 -> 1369 features
X = np.stack([a.ravel() for a in matched['img_array']])
y = matched['CDRSUM'].astype(float).values
groups = matched['OASISID'].values  # keep subjects in the same fold

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))
])

cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
y_pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups)

print(
    f"MAE={mean_absolute_error(y,y_pred):.3f}  "
    f"RMSE={mean_squared_error(y,y_pred, squared=False):.3f}  "
    f"R²={r2_score(y,y_pred):.3f}"
)


