import math
from collections import OrderedDict

import numpy as np


# modeling distance from i-th centroid to j-th centroid
class Distance:
    def __init__(self, start_index, end_index, euclidean_distance):
        self.start_index = start_index
        self.end_index = end_index
        self.euclidean_distance = euclidean_distance

    def __lt__(self, other):
        return self.euclidean_distance > other.euclidean_distance


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

        else:
            # computer centroid from bounding boxes
            input_centroids = np.zero(len(bounding_boxes), 2)

            for (i, (start_x, start_y, end_x, end_y)) in enumerate(bounding_boxes):
                # compute centroid for each bounding box
                cx = int((start_x + end_x) / 2 - 0)
                cy = int((start_y + end_y) / 2.0)
                input_centroids[i] = (cx, cy)

            # if no object has been tracked yet then
            if len(self.objects) == 0:
                for centroid in input_centroids:
                    self.register_object(centroid)

            # computer distances and update objects
            else:

                existing_object_ids = list(self.objects.keys())
                existing_centroids = list(self.objects.values())

                number_of_input_centroid = len(input_centroids)
                number_of_existing_centroids = len(existing_object_ids)

                # compute distance between new centroids and old existing ones:
                # TODO : improve time complexity of this part :
                for i in enumerate(input_centroids):
                    distances = []
                    for j in enumerate(existing_centroids):
                        euclidean_distance = self.compute_euclidean_distance(input_centroids[i], existing_centroids[j])

                        distance = Distance(i, j, euclidean_distance)
                        distances.append(distance)

                    # now sort distances from i-th input to existing centroid
                    distances = sorted(distances)
                    candidate = distances[0]

                    index = candidate.end_index
                    # update existing object centroid
                    self.objects[index] = input_centroids[i]
                    del input_centroids[i]
                    del existing_centroids[index]
                    del existing_object_ids[index]

                # check that if we have lost objects:
                # TODO : remove code duplication
                if number_of_existing_centroids >= number_of_input_centroid:
                    for id in existing_object_ids:
                        self.disappeared[id] += 1
                        # check if current object has to be considered lost
                        if self.disappeared[id] > self.max_disappeared:
                            self.unregister_object(id)
                else:
                    # register new objects
                    for centroid in input_centroids:
                        self.register_object(centroid)

        return self.objects

    def compute_euclidean_distance(self, c1, c2):
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(c1, c2)]))
        return distance
