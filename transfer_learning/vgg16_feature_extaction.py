from keras.applications import VGG16
from sklearn.preprocessing import LabelEncoder

from io.dataset_writer import HDF5DatasetWriter
from loaders.image_path_loader import load_image_paths


def extract_features_with_vgg16(dataset_path, batch_size=32, buffer_size=1000):
    paths, labels = load_image_paths(dataset_path, shuffle=True)

    # convers labels to vectors:
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    print('[INFO] LOADING VGG16 ')

    medel = VGG16(weights="imagenet", include_top=False)

    # initialize hdf5 writer:

    hdf5_dataset = HDF5DatasetWriter('./features.hdf5', len(paths), 512 * 7 * 7, buffer_size=buffer_size,
                                     dataset_name_keyword='features')

    hdf5_dataset.add()
