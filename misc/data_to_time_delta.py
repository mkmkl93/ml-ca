import os
import pandas as pd

from absl import flags
from absl import app
from tqdm import trange

flags.DEFINE_string('mode', 'train', 'In which mode should the program be run train/eval')
flags.DEFINE_string('source', '../data/uniform_200k/', 'Relative path to source data directory')
flags.DEFINE_string('result', '../data/uniform_200k_time_delta', 'Relative path to directory for results')

FLAGS = flags.FLAGS


def main(argv):
    source_dir = FLAGS.source
    result_dir = FLAGS.result

    for filename in os.listdir(source_dir):
        if filename.endswith('.csv'):
            dataset = pd.read_csv(source_dir + filename, index_col=0)
            (rows, columns) = dataset.shape

            for r in trange(0, rows):
                for c in reversed(range(0, columns - 1, 2)):
                    if dataset.iat[r, c] != 0:
                        dataset.iat[r, c] -= dataset.iat[r, c - 2]

            dataset.to_csv(result_dir + '/' + filename)


if __name__ == '__main__':
    app.run(main)
