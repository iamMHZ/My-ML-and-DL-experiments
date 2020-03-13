import cv2
import numpy  as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array


def augment(image, batch_size=1, save_image=False, output_path='./', total_number=20, save_prefix='image',
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2, zoom_range=0.2, horizontal_flip=True):

    # preprocess image :
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    augmentor = ImageDataGenerator(rotation_range=rotation_range, width_shift_range=width_shift_range,
                                   height_shift_range=height_shift_range, shear_range=shear_range,
                                   horizontal_flip=horizontal_flip, zoom_range=zoom_range)
    if save_image:
        image_generator = augmentor.flow(image, batch_size=batch_size, save_prefix=save_prefix, save_format='.jpg',
                                         save_to_dir=output_path)

    else:
        image_generator = augmentor.flow(image, batch_size=batch_size)

    # by every iteration of this loop , image_generator generates a new augmented data
    for i in range(total_number):
        augmented_image = image_generator.next()

        # display new image
        cv2.imshow('test', augmented_image)
        cv2.waitKey(50)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('./test.jpg')
    augment(img)
