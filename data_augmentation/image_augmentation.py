import os

'''
    simple image augmentor that flips, rotate and ...
    images of the script directory and saves them
'''


class SimpleAugmentor:

    def __init__(self, image_path='./', save_images=True, flip=True, gray=False, rotation=False,
                 filtering=False):
        self.save_images = save_images
        self.flip = flip
        self.gray = gray
        self.rotation = rotation
        self.filtering = filtering

        # if there is  directory for saving augmented data  then create one
        if not os.path.exists('plots'):
            os.makedirs('plots')

        self.image_path = image_path
        self.save_path = self.image_path + 'augments/'

    def start(self):
        pass

    def rotate_image(self, image):
        pass

    def flip_image(self, image):
        pass

    def convert_to_gray(self, image):
        pass

    def filer_image(self, image):
        pass

    def save_image(self, image, save_name):
        pass

    def load_images_form_disk(self):
        pass
