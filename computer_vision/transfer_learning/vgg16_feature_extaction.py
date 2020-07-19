import numpy as np
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder

from utils.hdf5.dataset_writer import HDF5DatasetWriter
from utils.loaders.image_path_loader import load_image_paths


def extract_features_with_vgg16(dataset_path, batch_size=32, buffer_size=1000):
    paths, labels = load_image_paths(dataset_path, shuffle=True)

    # convert labels to vectors:
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    print('[INFO] LOADING VGG16 ')

    model = VGG16(weights="imagenet", include_top=False)

    # initialize hdf5 writer:
    # we removed last FC layer of network and the output shape of POOL layer before this FC layer is  512 * 7 * 7
    hdf5_dataset = HDF5DatasetWriter('./features.hdf5', len(paths), 512 * 7 * 7, buffer_size=buffer_size,
                                     dataset_name_keyword='features')
    # save class labels in dataset
    hdf5_dataset.store_class_labels(label_encoder.classes_)

    print('[INFO] EXTRACTING FEATURES WITH NETWORK...')
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:batch_size]
        batch_labels = labels[i:batch_size]
        batch_images = []

        for path in batch_paths:
            image = load_img(path, target_size=(244, 244))
            image = img_to_array(image)

            # preprocess image for network:
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)

            batch_images.append(image)
            print('[INFO] FEATURE EXTRACTION...')

        if len(batch_images) > 0:
            batch_images = np.vstack(batch_images)
            # predicting features with VGG16 network:
            features = model.predict(batch_images, batch_size=batch_size)

            # writing row of data to hdf5 file
            features = features.reshape(features.shape[0], 512 * 7 * 7)
            hdf5_dataset.add(features, batch_labels)

    print('[INFO] DONE')
    hdf5_dataset.close()


if __name__ == '__main__':
    # dataset_path = 'D:\Programming\database of image\HAZMAT\Datasets\smallDataset'
    dataset_path = 'D:\Programming\database of image\Datasets\KaggleCatsAndDogs'

    extract_features_with_vgg16(dataset_path=dataset_path, batch_size=32, buffer_size=1000)
