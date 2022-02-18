from .base_hook import Hook


class CheckpointHook(Hook):
    """ CheckpointHook
    """
    def __init__(self, save_epochs):
        self.save_epochs = save_epochs

    def before_run(self, task):
        task.load_pretrain_model()

    def after_train_epoch(self, task, epoch):
        if epoch + 1 in self.save_epochs:
            task.save_ckpt(epoch + 1)
