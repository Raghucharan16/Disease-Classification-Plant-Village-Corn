import yaml, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from mmfnet.transforms import get_transforms
from mmfnet.datasets import ImageCSV
from mmfnet.models.rl_resnext import RLResNeXt
from sklearn.metrics import accuracy_score
import numpy as np, random

cfg = yaml.safe_load(open("configs/default.yaml"))
torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

train_csv, val_csv = "data/env/train.csv", "data/env/val.csv"
tfm_tr = get_transforms(cfg["rl"]["image_size"], train=True)
tfm_va = get_transforms(cfg["rl"]["image_size"], train=False)

train_ds = ImageCSV(train_csv, tfm_tr)
val_ds   = ImageCSV(val_csv, tfm_va)

train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

device = cfg["device"] if torch.cuda.is_available() else "cpu"
model = RLResNeXt(num_classes=4).to(device)
crit = nn.CrossEntropyLoss()
opt  = optim.Adam(model.parameters(), lr=cfg["lr"])

best = 0.0
for epoch in range(cfg["epochs"]):
    model.train()
    for img, y, _ in train_dl:
        img, y = img.to(device), y.to(device)
        opt.zero_grad()
        logits = model(img)
        loss = crit(logits, y)
        loss.backward(); opt.step()

    # validate
    model.eval(); preds=[]; gts=[]
    with torch.no_grad():
        for img, y, _ in val_dl:
            img = img.to(device)
            logits = model(img)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(y.tolist())
    acc = accuracy_score(gts, preds)
    print(f"epoch {epoch+1}/{cfg['epochs']}: val acc={acc:.4f}")
    if acc > best:
        best = acc
        torch.save(model.state_dict(), "checkpoints/rl_best.pth")
print("best:", best)
