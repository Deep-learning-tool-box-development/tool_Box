"""Convolutional Neural Network"""
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt
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

    def __init__(self, x_train, y_train, x_test, y_test):
        """
        default parameters:

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        """
        # print('Selected Network : CNN')
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train(self, params, plot=False):
        """
        Build and train the CNN, use the given parameters.

        :param params: list, [dropout, learning_rate, batch_size]
        :param plot: option, plot the learning curve
        :return: training history and the model for evaluate
        """
        # call the building function
        dropout, learning_rate, batch_size, conv = translate_params(params)
        model = build_network(dropout=dropout,
                              learning_rate=learning_rate,
                              num_conv=conv)
        history = model.fit(self.x_train,
                            self.y_train,
                            batch_size=batch_size,
                            epochs=12,
                            verbose=0,
                            validation_split=0.3)
        if plot:
            plot_learning_curve(history)
        return history, model

    def cnn_get_score(self, params):
        """
        Function to get the score of each model.

        :param params: list, [dropout, learning_rate, batch_size]
        :return: float, 1- mean value from last 3 validation accuracy
        """
        history, model = self.train(params)
        val_acc = history.history['val_accuracy']
        score = (val_acc[-3] + val_acc[-2] + val_acc[-1]) / 3
        return 1 - score

    def test(self, model):
        model.evaluate(self.x_test, self.y_test)
