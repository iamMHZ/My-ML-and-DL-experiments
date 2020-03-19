import cv2
import imutils
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

    def __init__(self, new_width, new_height, interpolation=cv2.INTER_AREA):
        self.width = new_width
        self.height = new_height

        self.inter = interpolation

    # resizing image and maintaining aspect ratio as well
    def pre_process(self, image):
        # old_height, old_width = image.shape[:2]
        #
        # print(old_height, old_width)
        #
        # aspect_ratio = old_width / old_height
        # print(aspect_ratio)
        #
        # final_width = int(self.new_height / aspect_ratio)
        # final_height = int(self.new_width / aspect_ratio)
        #
        # print(final_height, final_width)
        #
        # resized_with_aspect_ratio = cv2.resize(image, (final_width, final_height))
        # resized_without_aspect_ratio = cv2.resize(image, (self.new_width, self.new_height))
        #
        # # cropping from center:
        # difference_width = resized_with_aspect_ratio.shape[1] - self.new_width
        # difference_height = resized_with_aspect_ratio.shape[0] - self.new_height
        #
        # print(difference_height, difference_width)
        #
        # difference_height = int(difference_height / 2)
        # difference_width = int(difference_width / 2)
        #
        # final_result = resized_with_aspect_ratio[difference_height:self.new_height - difference_height,
        #                difference_width:self.new_width - difference_width]

        # cv2.imshow('original', image)
        # cv2.imshow('aspect_ratio', resized_with_aspect_ratio)
        # cv2.imshow('no aspect_ratio', resized_without_aspect_ratio)
        # cv2.imshow('final result', final_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        if w < h:
            image = imutils.resize(image, width=self.width,
                                   inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)

        else:
            image = imutils.resize(image, height=self.height,
                                   inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

            (h, w) = image.shape[:2]
            image = image[dH:h - dH, dW:w - dW]
            return cv2.resize(image, (self.width, self.height),
                              interpolation=self.inter)


if __name__ == '__main__':
    img = cv2.imread('./test2.jpg')

    preprocessor = AspectAwareResizePreprocessor(300, 500)

    preprocessor.pre_process(img)
