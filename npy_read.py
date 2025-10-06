import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

folder = "/Users/steventianlechen/Desktop/MSS Fall 2025/Research/barcodes_OASIS/G1/cs5_os3/CCs19_Cycles19/UNKNOWN/"
final_clinical_data = pd.read_csv("/Users/steventianlechen/Desktop/MSS Fall 2025/Research/final_clinical_data.csv")
images = []

files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
images = [np.load(os.path.join(folder, f)) for f in files]

