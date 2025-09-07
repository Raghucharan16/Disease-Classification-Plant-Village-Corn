# scripts/train_rl.py
from pathlib import Path
import sys

# --- Make project root importable ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from mmfnet.utils import set_seed, ensure_dir
from mmfnet.datasets import make_loaders
from mmfnet.models.rl_resnext import RLResNeXt

def main():
    cfg_path = ROOT / "configs" / "default.yaml"
    cfg = yaml.safe_load(open(cfg_path, "r"))

    import os, time, torch
    torch.set_num_threads(int(cfg.get("runtime", {}).get("threads", 8)))

    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    train_dir = str((ROOT / cfg["paths"]["train_dir"]).resolve())
    val_dir   = str((ROOT / cfg["paths"]["val_dir"]).resolve())
    ckpt_dir  = (ROOT / cfg["paths"]["ckpt_dir"]).resolve()
    ensure_dir(ckpt_dir)

    # dataloaders
    num_workers = int(cfg.get("num_workers", 0))
    _, _, dl_tr, dl_va = make_loaders(
        train_dir, val_dir,
        cfg["rl1"]["image_size"],
        cfg["batch_size"],
        num_workers,
    )

    # optional: use only a fraction for a quick smoke test (paper-faithful model/hparams)
    limit_pct = cfg.get("runtime", {}).get("limit_train_pct", None)
    if limit_pct:
        from torch.utils.data import Subset
        n = int(len(dl_tr.dataset) * float(limit_pct))
        dl_tr = torch.utils.data.DataLoader(
            Subset(dl_tr.dataset, range(n)),
            batch_size=cfg["batch_size"], shuffle=True, num_workers=num_workers
        )

    model = RLResNeXt(num_classes=4, pretrained=cfg["pl1"]["pretrained"]).to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    pe = int(cfg["runtime"]["print_every"])
    max_tb = cfg["runtime"]["max_train_batches"]
    max_vb = cfg["runtime"]["max_val_batches"]

    print(f"Train batches: {len(dl_tr)}  (items={len(dl_tr.dataset)})")
    print(f"Val   batches: {len(dl_va)}  (items={len(dl_va.dataset)})", flush=True)

    best_acc = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        running = 0.0
        for i, (xb, yb) in enumerate(dl_tr):
            xb, yb = xb.to(device, non_blocking=False), yb.to(device, non_blocking=False)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += float(loss.item())

            if ((i + 1) % pe == 0) or (i + 1 == len(dl_tr)):
                avg = running / (i + 1)
                print(f"[PL1][epoch {epoch}] batch {i+1}/{len(dl_tr)}  loss={avg:.4f}", flush=True)

            if max_tb and (i + 1) >= int(max_tb):
                break

        t1 = time.perf_counter()
        print(f"[PL1] epoch {epoch} train_time={t1 - t0:.1f}s", flush=True)

        # ----- validation -----
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for j, (xb, yb) in enumerate(dl_va):
                logits = model(xb.to(device))
                preds += logits.argmax(1).cpu().tolist()
                gts   += yb.tolist()
                if max_vb and (j + 1) >= int(max_vb):
                    break

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(gts, preds)
        print(f"[PL1] Epoch {epoch}/{cfg['epochs']}  val_acc={acc:.4f}", flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_dir / "pl1_best.pth")

    print(f"[PL1] best_val_acc={best_acc:.4f}", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()