import io
import neptune
import sys
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
from absl import flags
from data_processor import read_data

flags.DEFINE_list('filters', [32,32,32], 'Set list of CNN filters')
flags.DEFINE_float('dropout', 0.1, 'Set dropout')
flags.DEFINE_float('learning_rate', 0.0001, 'Set learning rate')
flags.DEFINE_string('activation', 'relu', 'Set activation method')
flags.DEFINE_string('data_dir', '../data/uniform_200k/', 'Relative path to the data folder')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs')
FLAGS = flags.FLAGS

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
                                   tags=["CNN", "grid", "10k_uniform", "1000_epochs", "data_v2", "mse"]):
        x_train, y_train, x_eval, y_eval = input_fn(trainingset_path)

        model = create_cnn(x_train.shape[1], 1,
                           filters=hparams["cnn_filters"],
                           dropout=hparams["dropout"],
                           activation=hparams["activation"],
                           kernel_size=hparams["kernel_size"]
                           )
        opt = Adam(lr=hparams["learning_rate"],
                   decay=1e-3 / 200)
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt)

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
        plt.close(figure)
        log_output(loss_history, hparams, path)


def id_from_hp(hp):
    return "cnn_kernel{}_filters{}_dropout{:.4f}_lr{}_{}".format(hp["kernel_size"], hp["cnn_filters"], hp["dropout"],
                                                                 hp["learning_rate"], hp["activation"])


def main(argv):
    neptune_tb.integrate_with_tensorflow()
    hparams = { "shuffle"                       : False,
                "num_threads"                   : 2,
                "batch_size"                    : 16384,
                "hidden_size"                   : 256,
                "hidden_units"                  : [512, 1024, 512, 256],
                "num_hidden_layers"             : 4,
                "initializer"                   : "uniform_unit_scaling",
                "initializer_gain"              : 1.0,
                "weight_decay"                  : 0.0,
                "l1_regularization_strength"    : 0.001,
                "kernel_size"                   : 9
            }
    session_num = 0
    hparams["cnn_filters"]      = [int(x) for x in FLAGS.filters]
    hparams["dropout"]          = FLAGS.dropout
    hparams["learning_rate"]    = FLAGS.learning_rate
    hparams["activation"]       = FLAGS.activation
    hparams['num_epochs']       = FLAGS.epochs
    dataDir                     = FLAGS.data_dir

    run_name = id_from_hp(hparams)
    print('--- Starting trial: %s' % run_name)
    train_and_eval(hparams, dataDir, dataDir + 'logs/')


if __name__ == '__main__':
    app.run(main)
