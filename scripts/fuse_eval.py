import yaml, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

from mmfnet.transforms import get_transforms
from mmfnet.models.rl_resnext import RLResNeXt
from mmfnet.models.pl_vgg16 import PLVGG16

def make_dl(root, size, bs, nw):
    ds = ImageFolder(root, transform=get_transforms(size, train=False))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    return ds, dl

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])

    # choose split to fuse on
    split_root = cfg["paths"]["val_dir"]  # change to test_dir for final test
    ds_rl, dl_rl   = make_dl(split_root, cfg["rl"]["image_size"],  cfg["batch_size"], cfg["num_workers"])
    ds_pl1, dl_pl1 = make_dl(split_root, cfg["pl1"]["image_size"], cfg["batch_size"], cfg["num_workers"])
    assert ds_rl.classes == ds_pl1.classes

    rl  = RLResNeXt().to(device); rl.load_state_dict(torch.load(ckpt_dir/"rl_best.pth",  map_location=device)); rl.eval()
    pl1 = PLVGG16().to(device);   pl1.load_state_dict(torch.load(ckpt_dir/"pl1_best.pth", map_location=device)); pl1.eval()

    # Option A: infer weights from standalone val accuracy (paper’s idea: weight by accuracy)
    if cfg["fusion"]["weights"] is None:
        # quick pass to compute per-stream acc
        def stream_acc(model, dl):
            preds, gts = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    preds += model(xb.to(device)).argmax(1).cpu().tolist()
                    gts   += yb.tolist()
            return accuracy_score(gts, preds)
        acc_rl  = stream_acc(rl, dl_rl)
        acc_pl1 = stream_acc(pl1, dl_pl1)
        w = np.array([acc_rl, acc_pl1], dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        print("weights from val acc -> RL, PL1:", w)
    else:
        w = np.array(cfg["fusion"]["weights"], dtype=np.float32)
        w = w / w.sum()

    preds, gts = [], []
    # IMPORTANT: both dls must have same ordering; keep shuffle=False and zip
    for (xb1, yb1), (xb2, yb2) in zip(dl_rl, dl_pl1):
        assert (yb1 == yb2).all(), "Label mismatch across streams; ensure identical splits & ordering"
        with torch.no_grad():
            p1 = F.softmax(rl(xb1.to(device)),  dim=1).cpu().numpy()
            p2 = F.softmax(pl1(xb2.to(device)), dim=1).cpu().numpy()
            p  = w[0]*p1 + w[1]*p2
            preds.extend(p.argmax(1).tolist())
            gts.extend(yb1.tolist())

    acc = accuracy_score(gts, preds)
    print("FUSION accuracy:", acc)
    print(classification_report(gts, preds, target_names=ds_rl.classes))
    print("Confusion matrix:\n", confusion_matrix(gts, preds))
