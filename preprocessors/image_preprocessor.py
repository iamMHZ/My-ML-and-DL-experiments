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


class AspectAwareResizePreprocessor:

    def __init__(self, new_width, new_height):
        self.new_width = new_width
        self.new_height = new_height

    # resizing image and maintaining aspect ratio as well
    def pre_process(self, image):
        old_height, old_width = image.shape[:2]

        aspect_ratio = old_width / old_height

        final_width = int(self.new_height / aspect_ratio)
        final_height = int(self.new_width / aspect_ratio)

        resized_with_aspect_ratio = cv2.resize(image, (final_width, final_height))
        resized_without_aspect_ratio = cv2.resize(image, (self.new_width, self.new_height))

        cv2.imshow('original', image)
        cv2.imshow('aspect_ratio', resized_with_aspect_ratio)
        cv2.imshow('no aspect_ratio', resized_without_aspect_ratio)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread('./test2.jpg')

    preprocessor = AspectAwareResizePreprocessor(300, 500)

    preprocessor.pre_process(image)
