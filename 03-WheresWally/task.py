"""
In this file you can call any of the tasks, train or predict with the model

You have just to change the FLAGS class to your needs

The output is a file result.csv with the result
"""

from src.model import train, predict
import sys

#TODO: pass these flags through a tensorflow app or via arguments
class FLAGS:

    #path to folder where the image dataset is found
    images_path = '/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/AugmentedImage/'
    
    # path with list of the images (read the README to know more about the csv)
    csv_path = '/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/AugmentedImage/list_images_shapes.csv'
    
    # after be trained the model will be saved, this is the path
    model_path = '/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/model_result/best_model.hdf5'
    
    # also a path to a  pickle file
    pickle_path = '/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/model_result/scaler.sav'
    
    input_shape = (128,128) #(width,height) input shape to the model
    batch_size = 16
    MAX_EPOCHS = 1000
    train_split = 0.9 # split on training or validating model


if __name__ == "__main__":

    help_message = "Run like this: python task.py `mode`\n\tthe mode can be train or predict.\n\tIf validation pass the path to the image"

    if len(sys.argv)<2:
        print(help_message)
    else:
        if sys.argv[1]=='train':
            train(FLAGS)
        elif sys.argv[1]=='predict' and len(sys.argv)>2:
            result = predict(FLAGS, sys.argv[2]).astype(int)
            with open("result.csv", 'w') as fr:
                file_name = sys.argv[2].split('/')[-1]
                fr.write(f'{file_name},{result[0][0]},{result[0][1]}')
        else:
            print(help_message)