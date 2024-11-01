import os.path

from torchvision import transforms as T
from torchvision.datasets import ImageFolder


def get_TinyImageNet(mode, data_root):
    if mode == "train":
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            T.Resize((224,224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])

        dataset = ImageFolder(os.path.join(data_root, "train"), transform)
    elif mode == "val":
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            normalize,
        ])

        dataset = ImageFolder(os.path.join(data_root, "val"), transform)
    return dataset
