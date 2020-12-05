# Solution for test 2 of Artificial Intelligence test

Hi there, this repository contains a solution for test 2.

To run this solution please: 

1. Install the requirements.txt in python enviroment or not
2. Download the model send via email and put it inside the folder 02-FraudDetection
3. Run the script in command line using:
	- python3 Classify_Signatures.py -m Siamese_model_all_data.hdf5 -img image_path_to_classify or 
		- The spected return is a print of type: "The class of this image is G"

	- python3 Classify_Signatures.py -m Siamese_model_all_data.hdf5 -dir-img directory_path_to_classify_images
		- The spected return is a print of type: "The class of image IMG_PATH is D" for each image in folder
4. Attention: 
	- This model PROBABLY NOT WILL WORK GOOD IN TESTSET the explanations why and the exaplanations of the choices for this model itself are in "Classify_Signatures.ipynb", to run just intantiate a jupyter notebook server
	- The directory_path_to_classify_images, has to be ONLY IMAGES to work well
	- The script was run it in Ubuntu 18.4.5 LST and with a 1050Ti GPU desktop
5. The model itself is the file "Siamese_model_all_data.hdf5" which is a only weights of a keras siamese model