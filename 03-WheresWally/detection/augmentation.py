import albumentations


def get_augmentation():
    aug = albumentations.Compose([
        albumentations.CLAHE(p=.2),
        albumentations.RandomSizedBBoxSafeCrop(500, 500, p=.2),
        albumentations.RandomBrightnessContrast(p=.2),
        albumentations.GaussNoise(p=.2),
        albumentations.HueSaturationValue(p=.2),
        albumentations.HorizontalFlip(p=.5),
        albumentations.VerticalFlip(p=.5),
    ], bbox_params=albumentations.BboxParams(format='pascal_voc', min_visibility=.3, label_fields=["labels", "iscrowd"]))

    return aug