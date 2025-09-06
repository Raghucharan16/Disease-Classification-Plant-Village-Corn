import yaml, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from mmfnet.utils import set_seed, ensure_dir
from mmfnet.datasets import make_loaders
from mmfnet.models.pl_vgg16 import PLVGG16
from sklearn.metrics import accuracy_score

cfg = yaml.safe_load(open("configs/default.yaml"))
set_seed(cfg["seed"])
device = cfg["device"] if torch.cuda.is_available() else "cpu"

train_dir = cfg["paths"]["train_dir"]
val_dir   = cfg["paths"]["val_dir"]
ckpt_dir  = Path(cfg["paths"]["ckpt_dir"]); ensure_dir(ckpt_dir)

_, _, dl_tr, dl_va = make_loaders(
    train_dir, val_dir, cfg["pl1"]["image_size"],
    cfg["batch_size"], cfg["num_workers"]
)

model = PLVGG16(num_classes=4, pretrained=cfg["pl1"]["pretrained"]).to(device)
crit  = nn.CrossEntropyLoss()
opt   = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

best_acc = 0.0
for epoch in range(1, cfg["epochs"]+1):
    model.train()
    for xb, yb in dl_tr:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward(); opt.step()

    model.eval(); preds=[]; gts=[]
    with torch.no_grad():
        for xb, yb in dl_va:
            out = model(xb.to(device)).argmax(1).cpu()
            preds += out.tolist(); gts += yb.tolist()
    acc = accuracy_score(gts, preds)
    print(f"[PL1] Epoch {epoch}/{cfg['epochs']}  val_acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), ckpt_dir/"pl1_best.pth")

print(f"[PL1] best_val_acc={best_acc:.4f}")
