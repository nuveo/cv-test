# Description

At the bottom of the sea? At the peak of the tallest mountain? Where might that Wally chap be? Is there a way to automatically determine Where's Wally?

A number of images with random backgrounds is provided. In each image, the same Wally picture was placed at a random position, with random rotation and a small perspective transform.
The following items are provided as `TrainSet` dataset:

- A collection of random images with Wally's picture at a random position
- The annotation json files (LabelMe standard), determining where Wally's picture is in each of the train images

The following items are provided as `TestSet` dataset:

- A collection of random images with Wally's picture at a random position

The following items are provided as `ReferenceData`:

- The original Wally picture
- A csv file containing the centroid of Wally's picture in each of the train images

# Objective

The objective of this test is to find a way to automatically detect Wally's picture in each of the `TestSet` images, giving the centroid of the picture in each image as the final answer.

# Important details

- Wally does not like being found. When asked about appearing in our test, he asked to personaly review the data first. We wonder if he tampered with the data to disturb our solutions...
- The dataset was split in order to have unseen data for test analysis. We took 20% of the total data (randomly)
- The annotations are in LabelMe standard. You can find the software [here](https://github.com/wkentaro/labelme)
- The CSV file contains the filename in the first column, the `x` position of the centroid in the second column and the `y` position of the centroid in the third column
- This test does not require a defined image processing algorithm to be used. The candidate is free to choose any kind of image processing pipeline to reach the best answer
- Depending on the chosen approach, not all provided files might be needed. We provide different resources so that different approaches are possible, but the candidate should feel free to use or discard any of the provided resources
- Replicate the data format for submission, i.e. the answer must be provided as a CSV file with the filename in the first column, the `x` coordinate for the centroid of the picture in the second column and the `y` coordinate for the centroid of the picture in the third column, similar to what is provided in the `TrainSet` dataset

# Solution

This problem could be solved in many ways. If running on low-end hardware was requisite, I believe haar-cascade would have been enough to detect, since it yields good results on rigid objects. Another approach would be to segment the polygon to obtain a higher precision centroid. In my solution, I decided to train a neural network, after all this is a test and I wanted to show this skill. The network I used was a Faster RCNN, which in my experience has dealt well with small datasets.

There are two important scripts in this test. "train.py" trains the CNN. The network converges fast and doesn't even trigger the lr scheduler. The "run_on_test.py" runs the inference in the test set.

# Important files

Besides the previously mentioned scripts, there are the following files:

- augmentation: Data augmentation
- transform: Data normalization and resize
- model: Loads the Faster RCNN from torchvision
- dataset: Loads the data and converts it to the format expected by the model
- coco_eval and coco_utils: Copied from the torchvision detection tutorial. Not my code
- inference: Interface for using the model for inference. Receives a BGR image, and return the boxes/scores in numpy format.

# About the data

To train the model the TrainingSet was divided into train and validation (20 images randomly chosen for validation). Also, a total of 8 images were removed because the annotations were "tampered by Wally".

# How to run

This project uses pycocotools, which has some issues with recent numpy versions, and therefore with recent opencv versions. The easiest way to run the code is to use the Dockerfile available.