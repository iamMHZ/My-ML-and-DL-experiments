import os
import random

import imutils


# load dataset's image paths and also shuffle them
# and return paths and labels
def load_data(dataset_path, shuffle=True):
    images_path = imutils.paths.list_images(dataset_path)

    if shuffle:
        random.shuffle(images_path)

    labels = []

    for path in images_path:
        label = path.split(os.path.sep)[-2]

        labels.append(label)

    return images_path, labels
