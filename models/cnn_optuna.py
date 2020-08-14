from keras.backend import clear_session
from keras.datasets import mnist
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
import optuna

import random
import io
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv1D, Activation, BatchNormalization, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from absl import flags, app
from datetime import datetime
from data_processor import read_data

flags.DEFINE_string('data_dir', '../data/uniform_200k/', 'Relative path to the data folder')
FLAGS = flags.FLAGS


BATCHSIZE = 16384
EPOCHS = 5

def input_fn(trainingset_path):
    x_train, y_train = read_data(trainingset_path, 'train')
    x_valid, y_valid = read_data(trainingset_path, 'eval')
    x_train = np.reshape(x_train.values, (-1, x_train.shape[1], 1))
    y_train = np.reshape(y_train.values, (-1, 1))
    x_valid = np.reshape(x_valid.values, (-1, x_valid.shape[1], 1))
    y_valid = np.reshape(y_valid.values, (-1, 1))
    return x_train, y_train, x_valid, y_valid


def create_cnn(trial, width, height):
    inputShape = (width, height)
    chanDim = -1

    num_of_conv_filters = trial.suggest_int("num_of_conv_filters", 1, 10)
    num_of_dense_filters = trial.suggest_int("num_of_dense_filters", 1, 5)
    kernel_size = trial.suggest_int("kernel_size", 1, 20)

    inputs = Input(shape=inputShape)
    x = inputs

    for i in range(num_of_conv_filters):
        x = Conv1D(
                filters=trial.suggest_int("conv_filter" + str(i), 1, 512),
                kernel_size=kernel_size,
                strides=trial.suggest_int("strides", 1, 20),
                activation="relu",
                padding="same"
            )(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(
                pool_size=trial.suggest_int("pool_size" + str(i), 1, 10),
                padding="same"
            )(x)

    x = Flatten()(x)

    for i in range(num_of_dense_filters):
        x = Dense(
                units=trial.suggest_int("dense_filter" + str(i), 1, 512),
            )(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(
                rate=trial.suggest_uniform("dense_dropout" + str(i), 0.0, 1.0)
            )(x)

    x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)

    return model


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    trainingset_path = FLAGS.data_dir
    x_train, y_train, x_valid, y_valid = input_fn(trainingset_path)

    model = create_cnn(trial, x_train.shape[1], 1)

    # We compile our model with a sampled learning rate.
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-1)
    decay = trial.suggest_loguniform("decay", 1e-6, 1e-1)
    opt = Adam(lr=lr, decay=decay)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(), optimizer=opt
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(x_valid, y_valid, verbose=0)
    return score


def main(argv):
    study = optuna.create_study(
        direction="minimize"
    )
    two_days_in_sec = 172800 - 10 * 60 # minus 10 min
    study.optimize(objective, n_trials=1, timeout=two_days_in_sec)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    df.to_csv('5000epochs.csv' + datetime.now().strftime("%H:%M:%S"))


if __name__ == '__main__':
    app.run(main)
