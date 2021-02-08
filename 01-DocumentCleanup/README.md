# Document Cleanup

## Description

The solution to this challenge is made up of two parts, one part of “traditional” image processing and the other involving character recognition techniques.

Initially, was tried to use only “traditional” image processing techniques, consisting of binary threshold, adaptive histogram equalization, gamma correction, filters, among others.
Due to the complexity of the problem, it was deemed necessary to use character recognition techniques to have a more “clean” final image and without the characters in italics.

A new version using other techniques to remove the italian formating that depends less on the pre-processing step could be implemented to obtain better results regardless of the initial noise levels.

The traditional image processing part consists of reading images from a specific folder, transforming the color space to gray scale, binarizing the image using a threshold and “automatic” rotation so that the text of the image is aligned with the horizontal plane.

In sequence, character recognition tools were used to extract the text from the processed images and generate a version without the noise from the original images.

The new images were exported to a separate folder, called output, keeping the same name pattern as the original images. 

## Usage
* Install tesseract from:

Source tesseract-ocr [Github](https://github.com/tesseract-ocr/tessdoc). <br>
Use on [Windows](https://medium.com/@ahmetxgenc/how-to-use-tesseract-on-windows-fe9d2a9ba5c6).


* Install requiriments

```
pip install -r requirements.txt
```