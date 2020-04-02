from keras.models import load_model

''' loading model with keras
    models with hdf5 extensions
'''


def load(model_path):
    model = load_model(model_path)

    return model
