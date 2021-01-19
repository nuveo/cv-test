# Description

Many image processing applications make use of digitalized textual data. However, the presence of any type of noise can create difficulties in post-processing information, such as on OCR detection. To improve the information manipulation on such data, a previous image processing step is required.

In light of this idea, a set of text paragraphs containing plain English language was collected. Different font styles, size, and background noise level were arranged to simulate the a variety of scenarios.

# Objective

The objective of this test is to evaluate the possible image processing methods that could fix the text samples. Note that the samples have a different type of background noise and present a set of text fonts. Therefore, the candidate should provide a flexible algorithm that can correctly detect what is text characters and background noise, offering a clean version of each text paragraph as result.

# Important details

- As a common pattern, the text must be represented by BLACK pixels and the background by WHITE pixels. Therefore, the output image MUST be in binary format (i.e. `0` pixel values for text and `255` pixel values for background)
- This test does not require a defined image processing algorithm to be used. The candidate is free to choose any kind of image processing pipeline to reach the best answer.
- The candidate will receive only the noisy data, as clean data is rarely provided on real-case scenarios, and no annotations are provided. Thus, creativity is needed if the candidate chooses to use supervised learning algorithms.
- The perfect correct result is reached with: 1) white background, 2) black text characters, 3) horizontal text aligment, 4) text block centered in the image and 5) straight text font (not itallic formatting).
- Do not change the filename when applying your image processing methods. The filename is important for data comparison purposes.
- The output image file will be only accepted on the following formats: `.png`, `.tif`, `.jpg`

# SOLUTION

The solution applied here was very simple. Since the text is usually darker than the background, an adaptative threshold was enough to remove most of the noise. The parameters of the adaptative threshold were selected empirically. To undo the skew, I used a function available [here](https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7). Finally, to remove some of the noise remaining between the lines, I looked for lines with less than 5% black pixels and changed the whole line to white. In my opinion, this step improved the visualization, although sometimes it removes the top or the bottom of some characters.

I tried to apply some low-pass filters but those removed the contrast between text and background and therefore didn't help. I also looked at the Fourier transform, but I didn't see anything helpful.

# How to run

All you have to do is run the script called "process_all.py". The only requirements are opencv and scipy.

# Tests

The repository also includes a few unit tests in a file called "tests.py".