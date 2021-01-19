# Where's Wally? Solution

## Requirements

| Python Library | Version |
| :---: | :---: |
| numpy | 1.18 |
| opencv-python | 4.4 |

## Solution

Since the original image was given, I decided to use a process of matching the reference with the query images.
The first idea that came to my mind was to use SIFT to find the keypoints of both images, a matcher (in code I used FLANN) to compute the corresponding points, and then homography to find the perspective transformation matrix.
This strategy worked beautifully and I was satisfied with the results.

## Running the code

```$ python3 find_wally.py <path to the reference image> <path to the query images> -e <image file extension> -o <output csv filename> --interactive --save-results
```

### Example

```$ python3 find_wally.py ReferenceData/wally.jpg TestSet -s```

This instruction will create a `output.csv` file with filenames and centroids, and a `results` folder containing images with Wally found.
