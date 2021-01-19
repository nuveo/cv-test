import albumentations
from albumentations.pytorch import ToTensorV2


def get_transform():
    """
    Returns a set of transforms, regularizes and transform images to tensors.
    :return: Callable set of transforms
    """
    transform = albumentations.Compose([
        albumentations.LongestMaxSize(max_size=500),
        albumentations.Normalize(mean=[0, 0, 0], std=(1, 1, 1)),
        ToTensorV2()
    ], bbox_params=albumentations.BboxParams(format='pascal_voc', min_visibility=.3, label_fields=["labels", "iscrowd"]))

    return transform