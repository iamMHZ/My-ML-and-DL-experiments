import os

import cv2
import numpy as np


class ImageLoader:

    def __init__(self, preprocessors=None):
        if preprocessors is None:
            preprocessors = []
        self.preprocessors = preprocessors

    def load(self, path):

        data = []
        labels = []

        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file_path)
                dir_name = os.path.basename(root)

                # file_path = file_path.replace('\\', '/')
                # label = file_path.split('/')[-2]
                # print(file_path)
                # print(dir_name)

                label = dir_name

                image = cv2.imread(file_path)

                # filter corrupted images
                if image is not None:
                    # pre_processing the image
                    for preprocessor in self.preprocessors:
                        image = preprocessor.pre_process(image)

                    print(file_path)
                    labels.append(label)
                    data.append(image)

        return np.array(data), np.array(labels)


if __name__ == "__main__":
    # path = 'D:\\Programming\\database of image\\Datasets\\KaggleCatsAndDogs\\'
    path = '../datasets/cats_and_dogs'
    loader = ImageLoader()
    loader.load(path)
