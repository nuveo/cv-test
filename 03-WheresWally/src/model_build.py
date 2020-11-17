from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError#, Huber
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
#import math

def cnn_model(FLAGS):
    """
    This function will build the model architecture that consist in a pre trained
    network which transfer its learning to another model.

    :param FLAGS: flags on the task.py file
    :return: builded model
    """

    base_model = MobileNetV2(input_shape=(FLAGS.input_shape[0], FLAGS.input_shape[1], 3),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False
    
    inputs = Input(shape = (FLAGS.input_shape[0], FLAGS.input_shape[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.21)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    output = Dense(2)(x)
    
    model = Model(inputs, output)

    return model


def compile_and_fit(model, train_ds, test_ds, MAX_EPOCHS, patience=20):
    """
    This function compile the model, fit and save the best fitted model in the filder model_result/best_model.hdf5.
    
    :param model: The model to be trained
    :param train_ds: A tensorflow dataset to train the model
    :param test_ds: A tensorflow dataset to test the model while training
    :param MAX_EPOCHS: num max of epochs to model be trained
    :param patience: the patience on finding the best this is, you are saying to the model "if the model doesn't
                     improve from this to the next $patience epochs stop the training"(default is 20)
    :return: History of training (you can see more about the history in the tensorflow documentation)
    """
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')

    save_model = ModelCheckpoint("model_result/best_model.hdf5",
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=2)

    #lr_schedule = LearningRateScheduler(lambda epoch:(0.5/(1+math.exp(-0.9*((epoch/30)-6))))/4)

    optimizer = Adam(learning_rate=16e-4)
    
    loss = MeanSquaredError()
    #loss = Huber()


    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[RootMeanSquaredError()])

    history = model.fit(train_ds, epochs=MAX_EPOCHS,
                        validation_data=test_ds,
                        callbacks=[save_model, early_stopping])

    return history