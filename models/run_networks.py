import os
from absl import app

def create_command():
    command = (
            "sbatch"
            + " --account=GR79-29"
            + " --nodes=1"
            + " --partition=gpu"
            + " --cpus-per-task=8"
            + " --time=48:00:00"
            + " --tasks-per-node=1"
            + " --gres=gpu:1"
    )
    return command

def grid_seach(command, program, dataDir, FILTERS, DROPOUTS, ACTIVATIONS, LEARNING_RATES):
    for filter in FILTERS:
        for dropout in DROPOUTS:
            for activation in ACTIVATIONS:
                for learningRate in LEARNING_RATES:
                    args = (
                        " --filters=" + ",".join(str(x) for x in filter)
                        + " --dropout=" + str(dropout)
                        + " --activation=" + activation
                        + " --learning_rate=" + str(learningRate)
                        + " --epochs=50000"
                        + " --data_dir=" + dataDir
                    )

                    script = "#!/bin/bash\npython3 " + program + args;
                    os.system("echo \"" + script + "\" | " + command)

def run_cnn():
    command = create_command()
    program = " ~/nasze-ca/models/cnn_regressor.py "

    filters = [[32, 32, 32, 32], [128, 128, 64, 32, 32]]
    dropout = [0.1, 0.2, 0.5]
    activation = ['relu']
    learnningRate = [0.0005, 0.0001]
    directory = "~/nasze-ca/data/uniform_200k/"

    grid_seach(command, program, directory, filters, dropout, activation, learnningRate)


def main(argv):
    run_cnn()


if __name__ == '__main__':
    app.run(main)
