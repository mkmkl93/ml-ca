import optuna
import optuna.integration.lightgbm as lgb
from sklearn.metrics import mean_squared_error
import neptune
import pandas as pd

import warnings

import numpy as np
warnings.filterwarnings("ignore")


def read_data(dataset_path, mode):
    print('Read dataset for ' + mode + ' from file ' + dataset_path + mode + '.csv')
    dataset = pd.read_csv(dataset_path + mode + '.csv')
    dataset = dataset.drop(['Unnamed: 0'], axis=1)
    return dataset.drop(['y'], axis=1), dataset['y']


import neptunecontrib.monitoring.optuna as opt_utils
neptune_monitor_optuna = opt_utils.NeptuneMonitor()


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print('Loading data...')
x_train, y_train = read_data('../data/', 'train')
x_eval, y_eval = read_data('../data/', 'eval')

# SET NEPTUNE_API_TOKEN in environment before running!
neptune.init('kowson/OLN')

used_params = []


print('Preparing LightGBM datasets...')
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_eval, y_eval, reference=lgb_train)


def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': ['gbdt'],
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'num_rounds': trial.suggest_int('num_rounds', 20, 300),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.6),
        'verbose': 0,
    }
    gbm = lgb.train(
        params=param,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        verbose_eval=False,
        early_stopping_rounds=5,
    )
    # PREDICT AND EVAL
    y_pred = gbm.predict(x_eval, num_iteration=gbm.best_iteration)
    #error = mean_squared_error(y_eval, y_pred) ** 0.5
    #print("RMSE of prediction is: {}".format(error))
    #neptune.log_text('rmse', str(error))
    error = mean_absolute_percentage_error(y_eval, y_pred)
    print("MAPE of prediction is: {}".format(error))
    neptune.log_metric('mape_error', error)
    #neptune.stop()
    return error


print("Optimizing...")
neptune.create_experiment(
    name='Optuna optimization with relative doses',
    tags=['optimization', 'optuna', 'lightgbm', 'data_v5', 'relative', 'uniform_200k', 'l2']
)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, callbacks=[neptune_monitor_optuna])


neptune.stop()
