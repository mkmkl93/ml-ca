import io
import neptune
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import neptune_tensorboard as neptune_tb

from tensorflow.keras.layers import Input, Conv1D, Activation, BatchNormalization, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from absl import flags
from absl import app
from datetime import datetime

from data_processor import read_data


def create_loss_figure(model, x, y):
    x_axis = np.arange(len(x))
    y_hat = model.predict(x)
    figure = plt.figure(figsize=(20, 10))
    plt.plot(x_axis, y_hat, 'r', label='predcited value')
    plt.plot(x_axis, y, 'b', label='actual value')
    plt.legend()
    return figure


def log_output(results, hparams, filepath):
    date = str(datetime.today().strftime('%Y%m%d%H%M%S'))
    filename = filepath + 'cnn_regressor_' + str(np.min(results['loss'])) + '_' + date + '.log'
    with open(filename, 'w') as f:
        f.write(str(hparams))
        f.write('\n\n')
        f.write(str(results))


def neptune_logs(figure):
    neptune.send_image("Loss on validation sample", figure)


def input_fn(trainingset_path):
    x_train, y_train = read_data(trainingset_path, 'train')
    x_eval, y_eval = read_data(trainingset_path, 'eval')
    x_train = np.reshape(x_train.values, (-1, x_train.shape[1], 1))
    y_train = np.reshape(y_train.values, (-1, 1))
    x_eval = np.reshape(x_eval.values, (-1, x_eval.shape[1], 1))
    y_eval = np.reshape(y_eval.values, (-1, 1))
    return x_train, y_train, x_eval, y_eval


def create_cnn(width, height, filters, dropout, activation, kernel_size):
    inputShape = (width, height)
    chanDim = -1

    inputs = Input(shape=inputShape)
    x = inputs
    for (i, f) in enumerate(filters):
        x = Conv1D(f, kernel_size, padding="same")(x)
        x = Activation(activation)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling1D(pool_size=(2))(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)

    x = Dense(64)(x)
    x = Activation("relu")(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)

    x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)

    return model


def train_and_eval(hparams, trainingset_path, model_path):
    path = model_path + id_from_hp(hparams)

    neptune.init(project_qualified_name='kowson/OLN')
    with neptune.create_experiment(name="Convolutional fully connected",
                                   params=hparams,
                                   tags=["CNN", "random", "150k_uniform", "1000_epochs"]):
        x_train, y_train, x_eval, y_eval = input_fn(trainingset_path)

        model = create_cnn(x_train.shape[1], 1,
                           filters=hparams["cnn_filters"],
                           dropout=hparams["dropout"],
                           activation=hparams["activation"],
                           kernel_size=hparams["kernel_size"]
                           )
        opt = Adam(lr=hparams["learning_rate"],
                   decay=1e-3 / 200)
        model.compile(loss="mean_squared_error", optimizer=opt)

        tbCallBack = callbacks.TensorBoard(
            log_dir=path, histogram_freq=1,
            write_graph=True, write_images=True)

        history_callback = model.fit(
            x_train,
            y_train,
            validation_data=(x_eval, y_eval),
            epochs=hparams["num_epochs"],
            batch_size=hparams["batch_size"],
            callbacks=[tbCallBack])

        loss_history = history_callback.history

        numOfExamples = 100
        figure = create_loss_figure(model, x_eval[:numOfExamples], y_eval[:numOfExamples])

        neptune_logs(figure)
        log_output(loss_history, hparams, path)


def id_from_hp(hp):
    return "cnn_kernel{}_filters{}_dropout{:.4f}_lr{}_{}".format(hp["kernel_size"], hp["cnn_filters"], hp["dropout"],
                                                                 hp["learning_rate"], hp["activation"])


def main():
    neptune_tb.integrate_with_tensorflow()
    HP_NUM_FILTERS = [[32,32,32,32], [128, 128, 64, 32, 32], [256, 256, 256, 256, 256]] 
    HP_DROPOUT = [0.0, 0.5, 0.1]
    HP_LEARNING_RATE = [0.0005, 0.001, 0.0001]
    HP_ACTIVATION = ['relu', 'tanh']
    HP_KERNEL_SIZE = [9, 11]
    hparams = { "num_epochs"                    : 1000,
                "shuffle"                       : False,
                "num_threads"                   : 2,
                "batch_size"                    : 16384,
                "hidden_size"                   : 256,
                "hidden_units"                  : [512, 1024, 512, 256],
                "num_hidden_layers"             : 4,
                "initializer"                   : "uniform_unit_scaling",
                "initializer_gain"              : 1.0,
                "weight_decay"                  : 0.0,
                "l1_regularization_strength"    : 0.001
            }
    session_num = 0
    for filters in HP_NUM_FILTERS:
        for dropout_rate in HP_DROPOUT:
            for activation in HP_ACTIVATION:
                for learning_rate in HP_LEARNING_RATE:
                    for kernel_size in HP_KERNEL_SIZE:
                        hparams["cnn_filters"] = filters
                        hparams["dropout"] = dropout_rate
                        hparams["learning_rate"] = learning_rate
                        hparams["activation"] = activation
                        hparams["kernel_size"] = kernel_size
                        run_name = id_from_hp(hparams)
                        print('--- Starting trial: %s' % run_name)
                        print(str(hparams))
                        session_num += 1
                        train_and_eval(hparams, '../data/', '../logs/')


if __name__ == '__main__':
    main()
