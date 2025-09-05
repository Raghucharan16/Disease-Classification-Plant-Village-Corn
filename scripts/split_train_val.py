import pandas as pd, os, shutil, numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

CSV = "data/env/env_records.csv"
OUT_CSV_TRAIN = "data/env/train.csv"
OUT_CSV_VAL   = "data/env/val.csv"
df = pd.read_csv(CSV)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=1337)
train_idx, val_idx = next(sss.split(df["image_path"], df["label"]))
df.iloc[train_idx].to_csv(OUT_CSV_TRAIN, index=False)
df.iloc[val_idx].to_csv(OUT_CSV_VAL, index=False)
print("train:", len(train_idx), "val:", len(val_idx))
