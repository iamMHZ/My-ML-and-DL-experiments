from keras.callbacks import BaseLogger


class TrainingMonitor(BaseLogger):
    '''
        serializing and plotting loss and accuracy of network
        for both training and validation set
    '''

    def __init__(self, plot_path, json_path=None, start_at=0):
        super(TrainingMonitor, self).__init__()

        self.plot_path = plot_path
        self.json_path = json_path
        self.start_at = start_at

        self.loss_history = {}

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass
