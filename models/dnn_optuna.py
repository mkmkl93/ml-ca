from keras.backend import clear_session
from keras.datasets import mnist
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
import optuna

import neptune
import random
import io
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import neptunecontrib.monitoring.optuna as opt_utils

from tensorflow.keras.layers import Input, Conv1D, Activation, BatchNormalization, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from absl import flags, app
from datetime import datetime
from neptunecontrib.monitoring.keras import NeptuneMonitor
from data_processor import read_data

flags.DEFINE_string('data_dir', '../data/uniform_200k/', 'Relative path to the data folder')
FLAGS = flags.FLAGS


BATCHSIZE = 65536
EPOCHS = 50000


def input_fn(trainingset_path):
    x_train, y_train = read_data(trainingset_path, 'train')
    x_valid, y_valid = read_data(trainingset_path, 'eval')
    x_train = np.reshape(x_train.values, (-1, x_train.shape[1], 1))
    y_train = np.reshape(y_train.values, (-1, 1))
    x_valid = np.reshape(x_valid.values, (-1, x_valid.shape[1], 1))
    y_valid = np.reshape(y_valid.values, (-1, 1))
    return x_train, y_train, x_valid, y_valid


def create_dnn(trial, width, height):
    inputShape = (width, height)

    num_of_dense_filters = trial.suggest_int("num_of_filters", 1, 8)
    dense_filter = trial.suggest_int("filter", 1, 512);
    dropout = trial.suggest_uniform("dropout", 0.1, 0.9);

    inputs = Input(shape=inputShape)
    x = inputs

    for i in range(num_of_dense_filters):
        x = Dense(
                units=dense_filter,
            )(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(
                rate=dropout,
            )(x)

    x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)

    return model


def objective(trial):
    clear_session()

    trainingset_path = FLAGS.data_dir
    x_train, y_train, x_valid, y_valid = input_fn(trainingset_path)

    model = create_dnn(trial, x_train.shape[1], 1)

    lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)
    decay = trial.suggest_loguniform("decay", 1e-6, 1e-4)
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
        callbacks=[NeptuneMonitor()],
        verbose=0
    )

    score = model.evaluate(x_valid, y_valid, verbose=0)
    return score


def main(argv):
    day_in_sec = 24 * 60 * 60 - 10 * 60  # minus 10 min

    neptune.init(project_qualified_name='kowson/OLN')
    neptune.create_experiment(name="DNN Optuna",
                              tags=["DNN", "optuna", "data_v2", "mse"])
    # neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)

    # study = optuna.create_study(
    #     study_name='dnn',
    #     direction="minimize",
    #     storage='sqlite:///cnn1.db',
    #     load_if_exists=True
    # )

    # study.optimize(objective, timeout=day_in_sec, callbacks=[neptune_callback])
    # opt_utils.log_study(study)

    # study.set_user_attr('n_epochs', EPOCHS)
    # print("Number of finished trials: {}".format(len(study.trials)))

    # print("Best trial:")
    # trial = study.best_trial
    #
    # print("  Value: {}".format(trial.value))
    #
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

    neptune.stop()

if __name__ == '__main__':
    app.run(main)
