class Hook(object):
    """ BaseClass for Hook
    """
    stages = ('before_run', 'before_train_epoch', 'before_train_iter',
        'after_train_iter', 'after_train_epoch', 'after_run')

    def before_run(self, *args):
        pass

    def before_train_epoch(self, *args):
        pass

    def before_train_iter(self, *args):
        pass

    def after_train_iter(self, *args):
        pass

    def after_train_epoch(self, *args):
        pass

    def after_run(self, *args):
        pass
