import cv2

from computer_vision.detection_and_tracking.centroid_object_tracking.tracker import CentroidTracker


def show_frame(frame, window_name, delay=1):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    cv2.waitKey(delay)


def main():
    camera = cv2.VideoCapture(0)
    face_cascade_path = './haarcascade_frontalface_default.xml'

    classifier = cv2.CascadeClassifier(face_cascade_path)

    centroid_tracker = CentroidTracker()

    while camera.isOpened():

        rectangles = []
        _, frame = camera.read()

        faces = classifier.detectMultiScale(frame)

        for face in faces:
            # print(faces)
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 120, 30))

            rectangles.append(face)

        objects = centroid_tracker.update(rectangles)

        for id, centroid in objects.items():
            cv2.circle(frame, center=centroid, radius=5, color=(0, 0, 200), thickness=-1)
            print(id, centroid, sep=' in ')

        show_frame(frame, 'frame')


if __name__ == '__main__':
    main()
