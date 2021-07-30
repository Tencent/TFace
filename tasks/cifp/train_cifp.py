import os
import sys
import logging
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

from tasks.localfc.train_localfc import TrainTask

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')


def main():
    """ main function with Traintask in localfc mode, which means each worker has a full classifier
    """
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainTask(os.path.join(task_dir, 'train_config.yaml'))
    task.init_env()
    task.train()


if __name__ == '__main__':
    main()
