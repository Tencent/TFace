import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))
from tasks.partialface.train import TrainTask

# parts of MinusFace's code requires the prior repository of PartialFace to sustain
# please refer to ../partialface for details

def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainTask(os.path.join(task_dir, 'train.yaml'))
    task.init_env()
    task.train()


if __name__ == '__main__':
    main()
