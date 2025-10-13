import re
import pandas as pd
import numpy as np
import os

folder = r"C:\Users\chent\PycharmProjects\Research\OASIS\barcodes_OASIS\G3\cs5_os3\CCs36_Cycles37\UNKNOWN"
clin_path = r"C:\Users\chent\PycharmProjects\Research\OASIS\final_clinical_data.csv"

# Clinical
clin = pd.read_csv(clin_path).rename(columns={'OASIS_session_label': 'clinical_session'})
clin['clinical_session'] = pd.to_numeric(clin['clinical_session'], errors='coerce')
clin['CDRSUM'] = pd.to_numeric(clin['CDRSUM'], errors='coerce')
clin = clin.dropna(subset=['OASISID', 'clinical_session', 'CDRSUM']).reset_index(drop=True)

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
# The above is for reading the .npy filenames.