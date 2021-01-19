import cv2
import pathlib as pl

import cleaner


def main():
    input_folder = pl.Path("noisy_data")
    output_folder = pl.Path("output")
    output_folder.mkdir(exist_ok=True)

    for fname in input_folder.glob("*.png"):
        print(fname)
        img = cv2.imread(str(fname))
        clean_img = cleaner.clean_document(img)
        cv2.imwrite(str(output_folder / fname.name), clean_img)


if __name__ == '__main__':
    main()