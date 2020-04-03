import pickle

import h5py
import numpy as np


# loads a model an features from hdf5 file and then calculates ranks for the models's predictions
def model_rank(model_path, hdf5_path):
    # load scikit-learn model
    model = pickle.load(open(model_path, 'rb').read())
    db = h5py.File(hdf5_path, 'r')

    index = db['labels'].shape[0] * 0.75

    # test model:
    predictions = model.predict_proba(db['features'][index:])

    rank1, rank5 = get_rank1_rank5(predictions, db['labels'][index:])

    print(f'RANK-1{rank1}')
    print(f'RANK-5{rank5}')


def get_rank1_rank5(predictions, ground_truth_labels):
    rank1 = 0
    rank5 = 0

    for (p, label) in zip(predictions, ground_truth_labels):

        # np.argsort() Returns the indices that would sort an array
        # in sorts array in ascending order
        p = np.argsort(p)

        # reverse the array order to be in descending order
        # so higher confident predictions will be at the front of the list
        p = p[::-1]

        if label in p[:5]:
            rank5 += 1

        if label == p[0]:
            rank1 += 1

    # computer final accuracies by :   correct/totals formula
    rank1 /= float(len(ground_truth_labels))
    rank5 /= float(len(ground_truth_labels))

    return rank1, rank5
