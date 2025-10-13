import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
# pd.set_option('display.max_columns', None)
# print(img_df)

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