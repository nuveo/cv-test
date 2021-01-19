import cv2
import numpy as np
import pathlib as pl
import torch
from tqdm import tqdm

from detection.coco_utils import get_coco_api_from_dataset
from detection.coco_eval import CocoEvaluator
from detection.dataset import LabelmeDataset
from detection.model import get_model


def main():
    model = get_model()

    max_epochs = 20

    train_dataest = LabelmeDataset("TrainingSet", augmentation=True)
    val_dataest = LabelmeDataset("ValidationSet", augmentation=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataest, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=train_dataest.collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataest, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=train_dataest.collate_fn)

    train_output = pl.Path("train_output")
    train_output.mkdir(exist_ok=True)

    count = 0

    for img, tgt in train_dataest:  # Visualize anns before training
        img_np = img.permute(1, 2, 0).numpy() * 255
        img_np = img_np[..., ::-1].astype(np.uint8)
        for b in tgt["boxes"]:
            img_np = cv2.rectangle(img_np, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

        cv2.imwrite(str(train_output / f"{count}.jpg"), img_np)
        count += 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode="max", verbose=True)

    model.to(device)

    best_map = 0

    n_thread = torch.get_num_threads()
    torch.set_num_threads(1)

    for epoch in range(1, max_epochs+1):
        model.train()
        for img, tgt in tqdm(train_loader):
            img = [i.to(device) for i in img]
            for t in tgt:
                for k in t.keys():
                    t[k] = t[k].to(device)

            losses = model(img, tgt)
            loss = sum(loss for loss in losses.values())
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        coco = get_coco_api_from_dataset(val_loader.dataset)
        coco_evaluator = CocoEvaluator(coco, ("bbox",))
        for images_th, targets in val_loader:
            images_th = list(image.to(device) for image in images_th)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = model(images_th, targets)

            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        epoch_map = coco_evaluator.coco_eval["bbox"].stats[0]
        lr_scheduler.step(epoch_map)

        if epoch_map > best_map:
            best_map = epoch_map
            torch.save(model.state_dict(), "best.pth")

        torch.set_num_threads(n_thread)

if __name__ == '__main__':
    main()
