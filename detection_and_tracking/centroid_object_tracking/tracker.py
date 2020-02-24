from collections import OrderedDict

import numpy as np


class CentroidTracker:

    def __init__(self, max_disappeared=50):
        # tracked objects
        self.objects = OrderedDict()
        # numbers of frames that an object is disappeared
        self.disappeared = OrderedDict()
        # next available ID for next object
        self.next_object_id = 0
        # maximum consecutive frames that after that we considered an object as disappeared
        self.max_disappeared = max_disappeared

    def register_object(self, centroid):
        # add new centroid to dictionary
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def unregister_object(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, bounding_boxes):
        # if there is no bounding box mark existing objects as disappeared
        if len(bounding_boxes) == 0:

            for id in list(self.objects.keys()):
                self.disappeared[id] += 1

                if self.disappeared[id] > self.max_disappeared:
                    self.unregister_object(id)

            return self.objects

        else:

            input_centroids = np.zero(len(bounding_boxes), 2)

            for (i, (start_x, start_y, end_x, end_y)) in enumerate(bounding_boxes):
                # compute centroid for each bounding box
                cx = int((start_x + end_x) / 2 - 0)
                cy = int((start_y + end_y) / 2.0)
                input_centroids[i] = (cx, cy)


