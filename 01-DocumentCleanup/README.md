# Document CleanUp Solution

## Requirements

| Python Library | Version |
| :---: | :---: |
| numpy | 1.18 |
| opencv-python | 4.4 |

## Solution

When dealing with text in non-uniform background (eg. because of light settings or images from photos), usually the first process is to apply an adaptive thresholding operation.
This first step will remove most of the background while affecting a small part of the text.
More often than not, we use mathematical morphology operations, eg. dilation and erosion, to further enhance text recognition.
However, after running experimental tests with morphology, I observed that a single adaptive threshold with a moderated block size and a large subtraction constant gave the best visual results.
Therefore, I decided to keep only this operation.

The next step was to align the text. 
First, I applied a median filter to remove some "pepper" noise (black spots).
Using OpenCV `minAreaRect`, I extracted the centroid, width, height, and angle of the best enclosing rectangle, which effectively tells where the text block is and how it is aligned.
After applying the warp affine function to adjust translation and rotation of the text, I had the final result.
Finally, the image is saved in the output folder.

## Running the code

```$ python3 document_cleanup.py <path to input images> -o <output folder name> -e <document file extension>
```

### Example

```$ python3 document_cleanup.py noisy_data -e .png```

This will create a `results` folder containing the images with segmented text.
