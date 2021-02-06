# Description

TrainingSet images were used to train a Faster rcnn inception v2 with inputs of 640x640 using Tensorflow's object detection API with no changes in standard COCO trained pipeline.config except batch size, labelmap and train/tes tfrecords.

The TrainingSet annotations were checked and wrong bounding boxes were corrected prior to tfrecord creation.

After 1000 steps total_loss metric was 0.07, checkpoint was created and used to export the model used.

Detection of Wally's pictures were done with this model.



# Usage

To use the inference model, please download and extract the folder found in the following link:

https://drive.google.com/file/d/1oJVB3E_WIcrlvyuntZTwOfB9u-RPrmu6/view?usp=sharing

Running find_wally.py processes all .jpg images in in --input_image_dir(default TestSet/) using model stored in --model_dir(default: FindWallyModel/saved_model/) with --threshold(default 0.6). Results are saved in output_image_dir(default output/) if --save is True(defautl) images with found bounding boxes are saved. Otherwise, only .csv file with the requested format is saved on the same output directory.

To rerun code on TestSet run find_wally.py without passing arguments.
