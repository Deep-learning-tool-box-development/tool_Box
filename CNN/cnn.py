"""Convolutional Neural Network"""
import os
import shutil
from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from utils import plot_learning_curve
from utils import translate_params


def build_network(dropout=0.5, learning_rate=0.004, num_conv=6):
    """
    Build the CNN network.

    :param num_conv:
    :param learning_rate:
    :param dropout:
    :return: CNN model built from given parameters
    """
    # build model from given parameters
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu',
                                  input_shape=(576, 1)))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu', padding="same"))
    if num_conv >= 4:
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding="same"))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding="same"))
    if num_conv >= 6:
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Conv1D(filters=256, kernel_size=2, activation='relu', padding="same"))
        model.add(keras.layers.Conv1D(filters=256, kernel_size=2, activation='relu', padding="same"))
    if num_conv >= 8:
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Conv1D(filters=512, kernel_size=1, activation='relu', padding="same"))
        model.add(keras.layers.Conv1D(filters=512, kernel_size=1, activation='relu', padding="same"))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(4, activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics='accuracy')
    return model


class CNN:

    def __init__(self, x_train, y_train,
                 x_test, y_test,
                 outdir=None,
                 logdir=None,
                 optimization=True,
                 epoch=5):
        """

        :param x_train: training data
        :param y_train: training label
        :param x_test: test data
        :param y_test: test label
        :param outdir: output directory
        :param logdir: log directory
        :param optimization: Bool, is it for optimization
        :param epoch: int, training epoch number
        """
        # print('Selected Network : CNN')
        self.optimization = optimization
        self.epoch = epoch
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None
        if self.optimization is False:
            if os.path.exists(outdir):
                shutil.rmtree(outdir)
            if os.path.exists(logdir):
                shutil.rmtree(logdir)
            os.makedirs(outdir)
            os.makedirs(logdir)

            self.outdir = outdir
            self.logdir = logdir

    def train(self, params, plot=False):
        """
        Build and train the CNN, use the given parameters.

        :param params: list, [dropout, learning_rate, batch_size]
        :param plot: bool, plot the learning curve
        :return: training history and the model for evaluate
        """
        # call the building function
        if self.optimization:
            dropout, learning_rate, batch_size, conv = translate_params(params)
        else:
            dropout = params[0]
            learning_rate = params[1]
            batch_size = params[2]
            conv = params[3]
        self.model = build_network(dropout=dropout,
                                   learning_rate=learning_rate,
                                   num_conv=conv)

        Early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=3,
                                                       mode='min')
        verbose = 1
        if self.optimization is False:
            verbose = 1
            checkpointer = keras.callbacks.ModelCheckpoint(filepath=self.outdir + "cnn_weights.hdf5",
                                                           # 给定checkpoint保存的文件名
                                                           monitor='val_accuracy',
                                                           mode='max',
                                                           verbose=verbose,
                                                           save_best_only=True)
            tensorboard = keras.callbacks.TensorBoard(log_dir=self.logdir)  # Tensorboard 保存地址是一个文件夹
            callbacks = [checkpointer, Early_stopping, tensorboard]
        else:
            callbacks = [Early_stopping]

        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 batch_size=batch_size,
                                 callbacks=callbacks,
                                 epochs=self.epoch,
                                 verbose=verbose,
                                 validation_split=0.3)
        if plot:
            plot_learning_curve(history)
        return history

    def cnn_get_score(self, params):
        """
        Function to get the score of each model.

        :param params: list, [dropout, learning_rate, batch_size]
        :return: float, 1- mean value from last 3 validation accuracy
        """
        history = self.train(params)
        val_acc = history.history['val_accuracy']
        score = (val_acc[-3] + val_acc[-2] + val_acc[-1]) / 3
        return 1 - score

    def test(self):
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print("Test accuracy:", test_acc)

    def save_model(self):
        """
        Save cnn model public Function
        """
        self.model.save(self.outdir + "cnn_model")

    def report(self):
        """
        Generating network report using test data

        :return: None
        """
        print(classification_report(np.argmax(self.y_test, axis=1),
                                    np.argmax(self.model.predict(self.x_test), axis=1),
                                    digits=4))

"""
Demo-code for CNN testing:
-------------------------------------------
x_train, x_test, y_train, y_test = import_data("./Dataset/", model='CNN')
cnn = CNN(x_train, y_train, x_test, y_test,
          outdir="./model/",
          logdir="./log/",
          optimization=False,
          epoch=10)
history = cnn.train([0.64, 0.004, 16, 6], plot=True)
cnn.report()
-------------------------------------------
Suggestion:
1. first argument for cnn.train should be: 
    [dropout, learning_rate, batch size, number of conv(4, 6, 8)]
2. try bigger batch_size
"""
