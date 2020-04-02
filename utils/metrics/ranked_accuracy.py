import numpy as np


def calculate_rank1_rank5(predictions, ground_truth_labels):
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
