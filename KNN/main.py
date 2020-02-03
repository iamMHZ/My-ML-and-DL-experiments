from KNN.image_loader import ImageLoader
from KNN.image_preprocessor import ImagePreprocessor
from KNN.classifier import trainKNN


def main():
    dataset_path = '../datasets/cats_and_dogs/'
    preprocessor = ImagePreprocessor(32, 32)
    loader = ImageLoader(preprocessor)

    (data, labels) = loader.load(dataset_path)
    data = data.reshape((data.shape[0], 3072))

    model = trainKNN(data, labels, 5)


if __name__ == '__main__':
    main()
