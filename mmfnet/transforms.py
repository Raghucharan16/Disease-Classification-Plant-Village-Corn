from torchvision import transforms

def get_transforms(sz, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(int(sz*1.15)),
            transforms.RandomResizedCrop(sz, scale=(0.7,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(sz),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
