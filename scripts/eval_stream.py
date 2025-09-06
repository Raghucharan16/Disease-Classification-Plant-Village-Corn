import yaml, torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mmfnet.datasets import make_loaders
from mmfnet.models.rl_resnext import RLResNeXt
from mmfnet.models.pl_vgg16 import PLVGG16

def eval_split(split="val", stream="rl"):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    img_size = cfg[stream]["image_size"]
    root = cfg["paths"][f"{split}_dir"]
    _, ds, _, dl = make_loaders(root, root, img_size, cfg["batch_size"], cfg["num_workers"])  # hack: same dir

    if stream == "rl":
        model = RLResNeXt().to(device); ckpt = "checkpoints/rl_best.pth"
    else:
        model = PLVGG16().to(device);   ckpt = "checkpoints/pl1_best.pth"

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb.to(device))
            preds += logits.argmax(1).cpu().tolist()
            gts   += yb.tolist()

    print(f"{stream.upper()} {split} acc:", accuracy_score(gts, preds))
    print(classification_report(gts, preds, target_names=ds.classes))
    print("Confusion matrix:\n", confusion_matrix(gts, preds))

if __name__ == "__main__":
    # python scripts/eval_stream.py  # tweak inside if needed
    eval_split("val", "rl")
    eval_split("val", "pl1")
