# Where's Wally?

## Description

The initial solution designed for this challenge was to use template matching functions at multiple scales to find Wally in the test images, however this method was very laborious and was not returning the expected results.

Due to the large amount of data available, it was decided to use neural networks to perform Wally's recognition. Using the Tensorflow object detection API, a Mobilenet v2 model was trained to perform inference on the images and report the central position of the bounding box found.

Some problems regarding the available label files were found, but due to the amount of images present on the test data, the images were lebeled again using the program labelImg, which would return the files in the desired formatting.

The result of the inferences and a file containing the central position of the bounding boxes can be found in the folder called output. 

## Usage
* Download, extract the Mobilenet v2 [model](https://drive.google.com/file/d/15QsE3-gjh16D8w3UCXUrbJshb5VkaEkY/view?usp=sharing) and place in the model folder.


* Install requiriments

```
pip install -r requirements.txt
```