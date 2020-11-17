from tensorflow.keras.models import load_model
import numpy as np

from src.gen_dataset import Dataset, gen_image_label, Predict
from src.model_build import cnn_model, compile_and_fit
from src.scripts.plot_history import plot_hist


#TODO: put the training history in a graphic to better visualization about what happened in the training
def train(FLAGS):
    """
    Training pipeline

    :param FLAGS: Flags values setted in the trask.py file
    """

    dataset = Dataset(FLAGS)
    train_data, test_data = dataset.train_test_split(train_split=FLAGS.train_split)

    train_data = train_data.map(lambda element, label:gen_image_label(element, label, [FLAGS.input_shape[0], FLAGS.input_shape[1]], FLAGS.images_path))
    train_data = dataset.configure(train_data, FLAGS.batch_size)

    test_data = test_data.map(lambda element, label:gen_image_label(element,label, [FLAGS.input_shape[0], FLAGS.input_shape[1]], FLAGS.images_path))
    test_data = dataset.configure(test_data, FLAGS.batch_size)

    model = cnn_model(FLAGS)
    history = compile_and_fit(model, train_data, test_data, FLAGS.MAX_EPOCHS)
    
    #save the training history of loss and validation
    plot_hist(history)


def predict(FLAGS, image_path):
    """
    prediction pipeline

    :param image_path: path to the image to be predicted
    :return: (x,y) points corresponding to the center of the object
    """

    data_process = Predict()
    data = data_process.gen_image(image_path, FLAGS.input_shape)

    model = load_model(FLAGS.model_path)
    result = model.predict(np.array([data,]))
    result = data_process.reverse_transform(result, FLAGS.pickle_path)

    return result
