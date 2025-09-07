from pathlib import Path
from collections import Counter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
paths = {
    "train": ROOT / "data" / "Train",
    "val":   ROOT / "data" / "Val",
    "test":  ROOT / "data" / "Test",
}
for split, p in paths.items():
    if not p.exists(): 
        print(f"[{split}] path missing:", p); 
        continue
    ds = ImageFolder(str(p), transform=transforms.ToTensor())
    counts = Counter(ds.targets)
    by_class = {ds.classes[k]: v for k, v in sorted(counts.items())}
    print(f"{split.upper()} total={len(ds)}")
    pprint(by_class)
