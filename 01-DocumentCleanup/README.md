# Document Cleanup

This is the algorithm to preprocessing a image before to use a OCR algorithm.


# How to run?  

1 - Install the Docker and run these commands:

- docker build . -t document_cleanup:v1
- docker run -it --name cont_1 document_cleanup:v1 

2 - **Run script**

- python main.py -p "noisy_data"


# Solution

The treatment of text images - in order for an OCR to detect characters - began with the reduction of noise in the image. For this, the Median Blur filter was used, which uses the neighbors' median to soften the image.
For the same purpose, an adaptive pixel intensity filter (by region) was used. The result of these two
filtering was joined in order to remove noise that appears only in one of the images.

The other concern was about the remaining noises (scratches, points) that were outside the boundary rectangle of the text. Since these noises would cause the rotation strategy used to fail. The rotation strategy is to determine the smallest rectangle that surrounds all points considered text. Once this is done, it is noticed when the rectangle is rotated in relation to the axis of the image itself. For the greater success of this rotation algorithm, a mask was made from expansion and erosion operations in order to represent only the rectangle corresponding to the text.
