import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(config, split="train"):
    """
    Returns data transforms. Uses strong online augmentations for the train split 
    to handle domain shift, and basic transforms for val/test.
    """
    image_size = config.get("data", {}).get("image_size", 224)
    mean = config.get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
    std = config.get("normalization", {}).get("std", [0.229, 0.224, 0.225])

    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 32), hole_width_range=(1, 32), fill=0, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
