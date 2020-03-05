import cv2
import numpy as np

from loaders import model_loader
from preprocessors.image_preprocessor import ResizePreprocessor, ImageToArrayPreprocessor


def main():
    image_to_array = ImageToArrayPreprocessor()
    resizer = ResizePreprocessor(32, 32)

    preprocessors = [resizer, image_to_array]

    # path to pre-trained model
    model_path = './shallowNetSign.hdf5'
    # load model
    model = model_loader.load(model_path)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _, frame = cap.read()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 32:
            cv2.destroyAllWindows()
            cap.release()
            break

        for preprocessor in preprocessors:
            frame = preprocessor.pre_process(frame)

        # predict data with model
        frame = np.expand_dims(frame, axis=0)

        predictions = model.predict(frame, batch_size=32).argmax(axis=1)

        # print class predicted class labels
        labels = ['Explosive', 'DWW', 'Radioactive']
        for pre in predictions:
            print(labels[pre])


if __name__ == '__main__':
    main()
