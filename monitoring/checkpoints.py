from keras.callbacks import ModelCheckpoint

'''
    A good application of checkpointing is to serialize your network to disk each time there is an
    improvement during training. We define an “improvement” to be either a decrease in loss or an
    increase in accuracy .
'''


def get_model_checkpoint_callback(file_name='weights', monitor='val_loss', mode='min', save_best_only=True, verboss=1):
    file_path = ''
    if monitor == 'val_loss':
        file_path = './' + file_name + '-weights-{epoch:03d}-{val_loss:.4f}.hdf5'

    elif monitor == 'val_acc':
        file_path = './' + file_name + '-weights-{epoch:03d}-{val_acc:.4f}.hdf5'

    checkpoint = ModelCheckpoint(filepath=file_path, monitor=monitor, mode=mode, save_best_only=save_best_only,
                                 verbose=verboss)

    return checkpoint
