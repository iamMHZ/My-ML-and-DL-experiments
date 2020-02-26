from loaders.image_loader import ImageLoader
from preprocessors.image_preprocessor import ResizePreprocessor
from KNN.classifier import trainKNN


def main():
    dataset_path = '../datasets/cats_and_dogs/'
    preprocessor = ResizePreprocessor(32, 32)
    loader = ImageLoader(preprocessor)

    (data, labels) = loader.load(dataset_path)
    data = data.reshape((data.shape[0], 3072))

    model = trainKNN(data, labels, 5)


if __name__ == '__main__':
    main()
