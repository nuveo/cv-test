import cv2
import pathlib as pl

from detection.inference import Inference


def main():
    inf = Inference("best.pth")
    test_folder = pl.Path("TestSet")
    output_folder = pl.Path("output")
    output_folder.mkdir(exist_ok=True)
    for img_fname in test_folder.glob("*"):
        img = cv2.imread(str(img_fname))
        result = inf.run_on_image(img)

        for b, s in zip(result["boxes"], result["scores"]):
            box_int = b.astype("int")
            img = cv2.rectangle(img, (box_int[0], box_int[1]), (box_int[2], box_int[3]), (0, 255, 0), 2)

            cv2.imwrite(str(output_folder / img_fname.name), img)


if __name__ == '__main__':
    main()