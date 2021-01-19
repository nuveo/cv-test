import cv2
import json
import numpy as np
import pathlib as pl
import torch
from torch.utils.data import Dataset

from .augmentation import get_augmentation
from .transform import get_transform


class LabelmeDataset(Dataset):
    def __init__(self, root_folder, augmentation=False):
        """
        Dataset initializer
        :param root_folder: Folder containing both jpg and json files in Labelme format
        :param augmentation: Boolean indicating whether or not transformations should be applied
        """
        super(LabelmeDataset, self).__init__()

        self.root_folder = pl.Path(root_folder)
        self.apply_aug = augmentation

        self.labels = list(self.root_folder.glob("*.json"))

        self.transform = get_transform()
        self.augmentation = get_augmentation()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with open(str(self.labels[index]), "r") as f:
            annotations = json.load(f)

        image_path = self.root_folder / annotations["imagePath"]

        image_np = cv2.imread(str(image_path))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        for a in annotations["shapes"]:
            points = np.asarray(a["points"])
            x0 = min(points[:, 0])
            y0 = min(points[:, 1])
            x1 = max(points[:, 0])
            y1 = max(points[:, 1])
            boxes.append([x0, y0, x1, y1])
            labels.append(1)

        iscrowd = np.zeros_like(labels)

        if self.apply_aug:
            data = self.augmentation(image=image_np, bboxes=boxes, labels=labels, iscrowd=iscrowd)
            boxes = data["bboxes"]
            image_np = data["image"]
            iscrowd = data["iscrowd"]
            labels = data["labels"]

        data = self.transform(image=image_np, bboxes=boxes, labels=labels, iscrowd=iscrowd)
        boxes = data["bboxes"]
        image = data["image"]
        iscrowd = data["iscrowd"]
        labels = data["labels"]
        areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]

        boxes = torch.from_numpy(np.asarray(boxes))
        iscrowd = torch.from_numpy(np.asarray(iscrowd))
        labels = torch.from_numpy(np.asarray(labels))
        areas = torch.from_numpy(np.asarray(areas))

        target = {
            "boxes": boxes,
            "iscrowd": iscrowd,
            "labels": labels,
            "area": areas,
            "image_id": torch.tensor(index).int()
        }

        return image, target

    def collate_fn(self, batch):
        images, targets = list(zip(*batch))
        return images, targets