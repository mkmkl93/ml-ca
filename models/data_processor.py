import os
import glob
import pandas as pd
import re

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from absl import flags
from absl import app
from tqdm import tqdm

flags.DEFINE_string('protocol_path', 'data/old_data/', 'Set a value for data source folder.')
flags.DEFINE_string('dataset_path', 'data/', 'Set a value for data destination folder.')
flags.DEFINE_integer('from_file', 1, 'Set a starting point for data processor.')
flags.DEFINE_integer('to_file', 33, 'Set an ending point for data processor.')
flags.DEFINE_boolean('concat_raw_data', True, 'Whether to read raw outputs first'
  'and only then preprocess for training')

FLAGS = flags.FLAGS


def process_protocols(path, from_file, to_file):
    all_proto = pd.DataFrame(0, index=[], columns=[i for i in range(0, 40)])
    extension = 'csv'
    row_i = 0
    print("Processing protocols")
    for i in range(from_file, to_file):
        all_filenames = [i for i in glob.glob(path + str(i) 
        	+ '/protocols/*.{}'.format(extension))]
        for f in tqdm(all_filenames, desc=str(i), ncols=100):
            temp = pd.read_csv(f, header=None)
            to_fill = (20 - temp.shape[1]) * 2  # fill first values with zeros
            row = [0] * to_fill
            for c in temp.columns:
                row = row + [temp.iloc[0, c], temp.iloc[1, c]]
            all_proto.at[row_i, :] = row
            row_i += 1
    return all_proto


def process_alive_counts(path, from_file, to_file):
    #'generated_final_good/data/data/'
    all_count = pd.DataFrame(0, index=[], columns=[0])
    extension = 'csv'
    row_i = 0
    print("Processing alive count")
    for i in range(from_file, to_file):
        all_filenames = [i for i in glob.glob(path + str(i) 
        	+ '/aliveCount/*.{}'.format(extension))]
        for f in tqdm(all_filenames, desc=str(i), ncols=100):
            temp = pd.read_csv(f)
            col_i = 0
            all_count.at[row_i, col_i] = temp.columns[0]
            row_i += 1
    return all_count    

def preprocess_data(dataset):
    dataset = dataset.astype(float).fillna(0)
    dataset['y'] = StandardScaler().fit_transform(
      dataset['y'].values.reshape(-1, 1))
    return dataset


def create_dataset_and_save(protocols, alive_counts, path, from_file, to_file):
    dataset = pd.concat([protocols, alive_counts], axis=1)
    dataset.columns = [str(i) for i in range (0, 40)] + ['y']
    dataset.to_csv(path + 'dataset' + str(from_file) + "_" + str(to_file) + '.csv')
    return dataset

def create_trainingset_and_save(dataset_path):
    dataset = pd.DataFrame()
    for file in os.listdir(dataset_path):
        if file.startswith("dataset") and file.endswith(".csv"):
            dataset = pd.concat(
              [dataset, pd.read_csv(file).drop(['Unnamed: 0'], axis=1)], 
              ignore_index=True)

    dataset = preprocess_data(dataset)

    x_train, x_infer, y_train, y_infer = train_test_split(
        dataset.drop(['y'], axis=1), dataset['y'], 
        test_size=0.2, random_state=1)

    x_infer, x_eval, y_infer, y_eval = train_test_split(
        x_infer, y_infer, test_size=0.5, random_state=1)
  
    pd.concat([x_train, y_train], axis=1).to_csv(dataset_path + 'train.csv')
    pd.concat([x_eval, y_eval], axis=1).to_csv(dataset_path + 'eval.csv')
    pd.concat([x_infer, y_infer], axis=1).to_csv(dataset_path + 'infer.csv')


def read_data(dataset_path, mode):
    print('Read dataset for ' + mode + ' from file ' + dataset_path + mode + '.csv')
    dataset = pd.read_csv(dataset_path + mode + '.csv')
    dataset = dataset.drop(['Unnamed: 0'], axis=1)
    return (dataset.drop(['y'], axis=1), dataset['y'])


def main(argv):
    print('Running with arguments...')
    print('Protocol path: ' + FLAGS.protocol_path)
    print('Dataset path: ' + FLAGS.dataset_path)
    if (FLAGS.concat_raw_data):
        print('Process starting with file: ' + str(FLAGS.from_file))
        print('Process ending with file: ' + str(FLAGS.to_file))
        protocols = process_protocols(FLAGS.protocol_path, FLAGS.from_file,
        	  FLAGS.to_file)
        alive_counts = process_alive_counts(FLAGS.protocol_path, FLAGS.from_file, 
        	FLAGS.to_file)
        dataset = create_dataset_and_save(protocols, alive_counts, FLAGS.dataset_path,
   	        FLAGS.from_file, FLAGS.to_file)
    else: 
        create_trainingset_and_save(FLAGS.dataset_path)

if __name__ == '__main__':
    app.run(main)
