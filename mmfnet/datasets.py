import pandas as pd, torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

CLASS2IDX = {"blight":0,"common_rust":1,"gray_leaf_spot":2,"healthy":3}

class ImageCSV(Dataset):
    def __init__(self, csv_path, tfm=None, sensor_cols=None, fit_scaler=None):
        self.df = pd.read_csv(csv_path)
        self.tfm = tfm
        self.sensor_cols = sensor_cols or []
        self.fit_scaler = fit_scaler
        if self.sensor_cols and self.fit_scaler is None:
            # fit min-max on this split to keep simple; for rigor, fit on train only and pass scaler to val
            self.mins = self.df[self.sensor_cols].min()
            self.maxs = self.df[self.sensor_cols].max().replace(0, 1)
        elif self.sensor_cols:
            self.mins, self.maxs = self.fit_scaler

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.image_path).convert("RGB")
        if self.tfm: img = self.tfm(img)
        y = torch.tensor(CLASS2IDX[r.label], dtype=torch.long)
        sensors = None
        if self.sensor_cols:
            x = (r[self.sensor_cols] - self.mins)/ (self.maxs - self.mins)
            sensors = torch.tensor(x.values.astype(np.float32))
        return img, y, sensors

    def get_scaler(self):
        return (self.mins, self.maxs)
