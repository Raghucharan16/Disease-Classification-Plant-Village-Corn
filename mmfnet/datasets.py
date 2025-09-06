from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .transforms import get_transforms

def make_loaders(train_dir, val_dir, img_size, batch_size, num_workers):
    ds_tr = ImageFolder(train_dir, transform=get_transforms(img_size, train=True))
    ds_va = ImageFolder(val_dir,   transform=get_transforms(img_size, train=False))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return ds_tr, ds_va, dl_tr, dl_va
