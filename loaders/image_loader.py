import os

import cv2
import numpy as np


class ImageLoader:

    def __init__(self, preprocessors=None):
        if preprocessors is None:
            preprocessors = []
        self.preprocessors = preprocessors

    def load(self, path, display_data=False):

        data = []
        labels = []

        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file_path)
                dir_name = os.path.basename(root)

                extension = file_name.split('.')[-1]

                label = dir_name

                # discarding other files:
                if extension == 'jpg' or extension == 'png':

                    try:

                        image = cv2.imread(file_path)
                        # display image if requested:
                        if display_data:
                            self.display(image, 'Dataset')

                        # pre_processing the image
                        for preprocessor in self.preprocessors:
                            image = preprocessor.pre_process(image)

                        print(file_path)
                        labels.append(label)
                        data.append(image)

                    except ValueError as val:
                        print(f'======> {val}')
                    except MemoryError:
                        print('======>  OUT OF MEMORY ')
                    except IOError:
                        print(f'======> ERROR IN LOADING : {file_name}')
                    except KeyboardInterrupt:
                        print(f'======> LOADING DATA STOPPED BY USER')

        cv2.destroyAllWindows()
        return np.array(data), np.array(labels)

    def display(self, image, window_name, delay=1):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)


if __name__ == "__main__":
    # path = 'D:\\Programming\\database of image\\Datasets\\KaggleCatsAndDogs\\'
    path = '../datasets/cats_and_dogs'
    loader = ImageLoader()
    loader.load(path,display_data=True)
