import cv2
from keras.preprocessing.image import img_to_array


class ResizePreprocessor:

    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def pre_process(self, image):
        # resizing the image
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)


# considers if images should be channel_first (channels, rows, columns) or channel_last (rows ,columns , channels)
class ImageToArrayPreprocessor:

    def __init__(self, data_format=None):
        self.data_format = data_format

    def pre_process(self, image):
        # keras function that changes the dimension of image
        # if self.data_format is None it means use whatever is in the keras.json file
        return img_to_array(image, data_format=self.data_format)
