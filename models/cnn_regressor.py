import neptune
import random
import sys
import numpy as np
import tensorflow as tf
import neptune_tensorboard as neptune_tb

from tensorflow.keras.layers import Input, Conv1D, Activation, BatchNormalization, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from absl import app, flags
from data_processor import read_data

flags.DEFINE_string('mode', 'train', 'In which mode should the program be run train/eval')
flags.DEFINE_list('cnn_filters', [32, 32, 32], 'List of CNN filters')
flags.DEFINE_list('cnn_pools', [8, 8, 8], 'List of pooling windows for each filter')
flags.DEFINE_list('dnn_filters', [128, 128], 'List of units of dense filters')
flags.DEFINE_list('dnn_dropout', [0.3, 0.3], 'List of dropouts for each dnn layer')
flags.DEFINE_integer('kernel', 10, 'Set kernel size')
flags.DEFINE_integer('stride', 1, 'Set stride size')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
flags.DEFINE_string('activation', 'relu', 'Activation method')
flags.DEFINE_string('data_dir', '../data/uniform_200k/', 'Relative path to the data folder')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs')
FLAGS = flags.FLAGS

RUN_NAME = 'run_{}'.format(random.getrandbits(64))
EXPERIMENT_LOG_DIR = 'logs/{}'.format(RUN_NAME)


def input_fn(trainingset_path):
    x_train, y_train = read_data(trainingset_path, 'train')
    x_eval, y_eval = read_data(trainingset_path, 'eval')
    x_train = np.reshape(x_train.values, (-1, x_train.shape[1], 1))
    y_train = np.reshape(y_train.values, (-1, 1))
    x_eval = np.reshape(x_eval.values, (-1, x_eval.shape[1], 1))
    y_eval = np.reshape(y_eval.values, (-1, 1))
    return x_train, y_train, x_eval, y_eval


def create_cnn(input_shape, hparams):
    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(len(hparams["cnn_filters"])):
        x = Conv1D(filters=hparams['cnn_filters'][i],
                   kernel_size=hparams['kernel'],
                   padding="same",
                   strides=hparams['stride'])(x)
        x = Activation(hparams['activation'])(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=hparams['cnn_pools'][i],
                         padding='same')(x)

    x = Flatten()(x)

    for i in range(len(hparams['dnn_filters'])):
        x = Dense(units=hparams['dnn_filters'][i])(x)
        x = Activation(hparams['activation'])(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=hparams['dnn_dropout'][i])(x)

    x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)

    return model


def train(hparams, trainingset_path):
    neptune.init(project_qualified_name='kowson/OLN')
    with neptune.create_experiment(name="Convolutional fully connected",
                                   params=hparams,
                                   tags=["CNN", "grid", "10k_uniform", "1000_epochs", "data_v2", "mse"]):
        x_train, y_train, x_eval, y_eval = input_fn(trainingset_path)

        input_shape = (x_train.shape[1], 1)
        model = create_cnn(input_shape, hparams)
        opt = Adam(lr=hparams["learning_rate"],
                   decay=1e-3 / 200)

        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt)

        tbCallBack = callbacks.TensorBoard(log_dir=EXPERIMENT_LOG_DIR)

        checkpoint_path = EXPERIMENT_LOG_DIR
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        model.fit(x_train,
                  y_train,
                  validation_data=(x_eval, y_eval),
                  epochs=hparams["num_epochs"],
                  batch_size=hparams["batch_size"],
                  callbacks=[tbCallBack])


def check_hparams(hparams):
    if len(hparams["cnn_filters"]) != len(hparams["cnn_pools"]):
        return 1

    if len(hparams["dnn_filters"]) != len(hparams["dnn_dropout"]):
        return 1


def create_experiment():
    neptune_tb.integrate_with_tensorflow()
    hparams = {'shuffle': False,
               'num_threads': 2,
               'batch_size': 16384,
               'num_hidden_layers': 4,
               'initializer': "uniform_unit_scaling",
               'initializer_gain': 1.0,
               'weight_decay': 0.0,
               'l1_regularization_strength': 0.001,
               'cnn_filters': [int(x) for x in FLAGS.cnn_filters],
               'cnn_pools': [int(x) for x in FLAGS.cnn_pools],
               'dnn_filters': [int(x) for x in FLAGS.dnn_filters],
               'dnn_dropout': [float(x) for x in FLAGS.dnn_dropout],
               'kernel': FLAGS.kernel,
               'stride': FLAGS.stride,
               'learning_rate': FLAGS.learning_rate,
               'activation': FLAGS.activation,
               'num_epochs': FLAGS.epochs
               }

    dataDir = FLAGS.data_dir

    if check_hparams(hparams):
        print("Incorrect hyper parameters")
        sys.exit(1)

    print('--- Starting trial ---')
    train(hparams, dataDir)


def main(argv):
    if FLAGS.mode == 'train':
        create_experiment()
    elif FLAGS.mode == 'eval':
        pass
    else:
        print("Incorrect mode")
        sys.exit(1)


if __name__ == '__main__':
    app.run(main)
