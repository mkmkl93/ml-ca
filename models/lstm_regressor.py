import random
import neptune
import numpy as np
import tensorflow as tf
import neptune_tensorboard as neptune_tb

from absl import flags
from absl import app
from data_processor import read_data
from tensorflow.keras import callbacks, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

flags.DEFINE_string('mode', 'train', 'In which mode should the program be run train/eval')
flags.DEFINE_float('dropout', 0.2, 'Set dropout')
flags.DEFINE_float('learning_rate', 0.0001, 'Set learning rate')
flags.DEFINE_string('activation', 'relu', 'Set activation method')
flags.DEFINE_string('data_dir', '../data/uniform_200k_time_delta/', 'Relative path to the data folder')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_integer('lstm_filters', 1, 'Number of LSTM layers')
flags.DEFINE_integer('lstm_filter', 256, 'Number of units in lstm')
flags.DEFINE_integer('dense_filters', 1, 'Number of Dense layers')
flags.DEFINE_integer('dense_filter', 256, 'Number of units in dense layer')
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


def create_lstm(hparams):
    model = Sequential()
    model.add(BatchNormalization())
    
    for _ in range(hparams['lstm_filters']):
        model.add(LSTM(hparams['lstm_filter'], return_sequences=True))
    model.add(LSTM(hparams['lstm_filter']))

    for _ in range(hparams['dense_filters']):
        model.add(Dense(hparams['dense_filter'], activation='relu'))
        model.add(Dropout(hparams['dropout']))
        model.add(BatchNormalization())

    return model

def divide(x):
    ret = np.empty([len(x), 20, 2])
    for i, row in enumerate(x):
        for j in range(1, len(row)):
            j -= 1
            if (j - 1) % 2:
                ret[i][(j - 1) // 2][0] = row[j + 1]
            else:
                ret[i][(j - 1) // 2][1] = row[j + 1]
    
    return ret

def train(hparams):
    neptune.init(project_qualified_name='kowson/OLN')
    with neptune.create_experiment(name="Recurrent neural network",
                                   params=hparams,
                                   tags=["CuDNNLSTM", hparams['data_dir'], "data_v2", "mse"]):
        x_train, y_train, x_eval, y_eval = input_fn(hparams['data_dir'])
        x_train_divided = divide(x_train)
        x_eval_divided = divide(x_eval)
        
        model = create_lstm(hparams)
        opt = Adam(lr=hparams["learning_rate"],
                   decay=1e-3 / 200)
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt)

        tbCallBack = callbacks.TensorBoard(log_dir=EXPERIMENT_LOG_DIR)

        checkpoint_path = EXPERIMENT_LOG_DIR
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,

                                                         save_freq=11*100,
                                                         verbose=1)

        history_callback = model.fit(
            x_train,
            y_train,
            validation_data=(x_eval, y_eval),
            epochs=hparams["num_epochs"],
            batch_size=hparams["batch_size"],
            callbacks=[tbCallBack, cp_callback],
            verbose=2)


def create_experiment():
    neptune_tb.integrate_with_tensorflow()
    hyper_params = {
            'data_dir': FLAGS.data_dir,
            'shuffle': False,
            'num_threads': 1,
            'batch_size': 16384,
            'initializer': 'uniform_unit_scaling',
            'lstm_filters': FLAGS.lstm_filters,
            'lstm_filter': FLAGS.lstm_filter,
            'dense_filters': FLAGS.dense_filters,
            'dense_filter': FLAGS.dense_filter,
            'dropout': FLAGS.dropout,
            'learning_rate': FLAGS.learning_rate,
            'activation': 'relu',
            'num_epochs': FLAGS.epochs,
    }

    print('--- Starting trial ---')
    train(hyper_params)


def main(argv):
    if FLAGS.mode == 'train':
        create_experiment()
    elif FLAGS.mode == 'eval':
        pass


if __name__ == '__main__':
    app.run(main)

