import json
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import BaseLogger


class TrainingMonitor(BaseLogger):
    '''
        serializing and plotting loss and accuracy of network
        for both training and validation set
    '''

    def __init__(self, plot_name, json_path=None, start_at=0, save_plots=False):
        super(TrainingMonitor, self).__init__()

        self.plot_name = plot_name

        # if there is no plot directory for saving plots then create one
        if not os.path.exists('plots'):
            os.makedirs('plots')

        self.plot_path = f'./plots/{self.plot_name}'

        self.json_path = json_path
        self.start_at = start_at
        self.save_plots = save_plots

        self.loss_history = {}

    def on_train_begin(self, logs=None):
        if self.json_path is not None:
            pass

    def on_epoch_end(self, epoch, logs=None):

        if logs is not None:

            for (key, value) in logs.items():
                l = self.loss_history.get(key, [])
                l.append(value)
                self.loss_history[key] = l

        # serialize to json file:
        if self.json_path is not None:
            file = open(self.json_path, 'w')

            file.write(json.dumps(self.loss_history))
            file.close()

        # plotting
        if self.plot_path is not None:
            # plot the training loss and accuracy

            if len(self.loss_history['loss']) > 1:
                print('[INFO] plotting...')

                # epoch+1 because np.arange()  does not include stop value
                x = np.arange(start=0, stop=epoch + 1)

                plt.style.use("ggplot")
                plt.figure()
                plt.plot(x, self.loss_history["loss"], label="train_loss")
                plt.plot(x, self.loss_history["val_loss"], label="val_loss")
                plt.plot(x, self.loss_history["accuracy"], label="train_acc")
                plt.plot(x, self.loss_history["val_accuracy"], label="val_acc")
                plt.title("Training Loss and Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Loss/Accuracy")
                plt.legend()

                if self.save_plots:
                    plt.savefig(self.plot_path + str(epoch) + '.png')
                plt.show()
