import os

import cv2

'''
    simple image augmentor that flips, rotate and ...
    images of the script directory and saves them
'''


class SimpleAugmentor:

    def __init__(self, image_path='./', save=True, flip=True, rotation=True, gray=False,
                 filtering=False):
        self.save = save
        self.flip = flip
        self.gray = gray
        self.rotation = rotation
        self.filtering = filtering

        # if there is  directory for saving augmented data  then create one
        if not os.path.exists('augments'):
            os.makedirs('augments')

        self.image_path = image_path
        self.save_path = self.image_path + '/augments/'

    # call  this method to start augmenting data
    def start(self):

        images, names = self.load_images_form_disk()

        for image, name in zip(images, names):

            if self.rotation:
                self.rotate_image(image, name)

            if self.flip:
                self.flip_image(image, name)

            if self.gray:
                self.convert_to_gray(image, name)

            if self.filtering:
                self.filter_image(image, name)

    def rotate_image(self, image, name):
        rotate_codes = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]

        for i, code in enumerate(rotate_codes):
            rotated = cv2.rotate(image, code)
            if self.save:
                self.save_image(rotated, save_name=name + 'Rotate' + str(i))

    def flip_image(self, image, name):

        for i in range(0, 2):
            flipped = cv2.flip(image, flipCode=i)
            self.save_image(flipped, name + 'Flip' + str(i))

    def convert_to_gray(self, image, name):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.save_image(gray, 'gray ' + name)

    def filter_image(self, image, name):
        # decide filtering type
        pass

    def save_image(self, image, save_name):
        cv2.imwrite(self.save_path + save_name + '.jpg', image)
        print(f'[INFO] saved {save_name}')

    def load_images_form_disk(self):
        images = []
        names = []

        for root, dirs, files in os.walk(self.image_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file_path)

                # remove extension
                file_name = file_name[:-4]

                image = cv2.imread(file_path)

                # filter corrupted images
                if image is not None:
                    images.append(image)
                    names.append(file_name)

        return images, names


if __name__ == '__main__':
    augmentor = SimpleAugmentor()

    augmentor.start()
