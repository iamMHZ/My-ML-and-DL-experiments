#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install tensorflow==1.14.0
import warnings
import dlib
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import cv2
from keras.models import Model
import os
import glob
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,Dropout,Activation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from numpy import asarray
from numpy import clip
from PIL import Image
get_ipython().system('pip install git+https://github.com/rcmalli/keras-vggface.git')
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn import svm


# In[2]:


get_ipython().system('mkdir /content/home')
get_ipython().system('mkdir /content/home/train')
get_ipython().system('mkdir /content/home/test')
get_ipython().system('mkdir /content/home1')
get_ipython().system('mkdir /content/home1/train')
get_ipython().system('mkdir /content/home1/test')


# In[4]:


import zipfile
zip_ref = zipfile.ZipFile('/content/drive/My Drive/Freelance/LFW_Keras/lfw_funneled.zip', 'r')
zip_ref.extractall('/content/home/train')
zip_ref.close()


# In[5]:


get_ipython().system('pip install mtcnn')
get_ipython().system('pip install keras-facenet')
get_ipython().system('pip install imutils tqdm')
get_ipython().system('pip install tqdm')


# In[6]:


import base64
import glob
import os
import time
import warnings

import cv2
import imutils
import math
import numpy as np
import tensorflow as tf
import tqdm
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN


# In[7]:


FACE_SIZE = (160, 160)
INPUT_SHAPE = (160, 160, 3)
FACE_DETECTION_SIZE = (240, 200)
MIN_FACE_SIZE = (48, 48)
UNKNOWN_LABEL = "UNKNOWN"
FACE_CONFIDENCE = .9
DEFAULT_THRESHOLDS = [0.6, 0.9]
ALLOWED_IMAGE_TYPES = ('*.jpeg', '*.png', '*.jpg')


# In[47]:


class SVMCLASSIFIRE(object):


    def __init__(self):

        self.X = []
        self.Y = []

    def fit(self, x_train, y_train):

        if len(self.X) > 0 and len(self.Y) > 0:
            self.X = np.concatenate((self.X, x_train), axis=0)
            self.Y = self.Y + y_train
        else:
            self.X = x_train
            self.Y = y_train

    def predict(self, x_test):
        clf = svm.SVC(C=7.0, kernel='rbf', degree=9, gamma='auto', coef0=0.0,
                              shrinking=True, probability=False, tol=0.0001, cache_size=200,
                              verbose=False, max_iter=-1, decision_function_shape='ovr',
                              break_ties=False, random_state=12)
        clf.fit(self.X, self.Y)
        result = clf.predict(np.expand_dims(x_test, axis=0))
        idx = self.Y.index(result[0])
        max_val, min_val = 1, -1
        raw_confidence = max(1, min_val)
        return {"person": self.Y[idx], "confidence": (raw_confidence - min_val) / (max_val - min_val)}

    def load(self, path):

        database = pickle.load(open(path, "rb"))

        self.X = database["encodings"]
        self.Y = database["people"]

    def save(self, path):


        database = {
            "encodings": self.X,
            "people": self.Y
        }

        pickle.dump(database, open(path, "wb"))


# In[29]:


from sklearn.metrics.pairwise import cosine_similarity
class Cosine_Similarity(object):


    def __init__(self):

        self.X = []
        self.Y = []

    def fit(self, x_train, y_train):

        if len(self.X) > 0 and len(self.Y) > 0:
            self.X = np.concatenate((self.X, x_train), axis=0)
            self.Y = self.Y + y_train
        else:
            self.X = x_train
            self.Y = y_train

    def predict(self, x_test):

        similarity = cosine_similarity(np.expand_dims(x_test, axis=0), self.X)[0]
        idx = int(np.argmax(similarity))
        max_val, min_val = 1, -1
        raw_confidence = max(similarity[idx], min_val)
        return {"person": self.Y[idx], "confidence": (raw_confidence - min_val) / (max_val - min_val)}

    def load(self, path):

        database = pickle.load(open(path, "rb"))

        self.X = database["encodings"]
        self.Y = database["people"]

    def save(self, path):


        database = {
            "encodings": self.X,
            "people": self.Y
        }

        pickle.dump(database, open(path, "wb"))


# In[8]:


import pickle

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class EuclideanClassifier(object):


    def __init__(self):

        self.X = []
        self.Y = []

    def fit(self, x_train, y_train):

        if len(self.X) > 0 and len(self.Y) > 0:
            self.X = np.concatenate((self.X, x_train), axis=0)
            self.Y = self.Y + y_train
        else:
            self.X = x_train
            self.Y = y_train

    def predict(self, x_test):

        distances = euclidean_distances(np.expand_dims(x_test, axis=0), self.X)[0]
        idx = int(np.argmin(distances))
        max_val, min_val = 1, -1
        raw_confidence = max(1 - distances[idx], min_val)
        return {"person": self.Y[idx], "confidence": (raw_confidence - min_val) / (max_val - min_val)}

    def load(self, path):

        database = pickle.load(open(path, "rb"))

        self.X = database["encodings"]
        self.Y = database["people"]

    def save(self, path):


        database = {
            "encodings": self.X,
            "people": self.Y
        }

        pickle.dump(database, open(path, "wb"))


# In[9]:



ROOT_FOLDER ="/content/home/train/lfw_funneled"
MODEL_PATH = "lfw_model.pkl"


# In[10]:


dataset = []
for path in glob.iglob(os.path.join(ROOT_FOLDER, "**", "*.jpg")):
    person = path.split("/")[-2]
    dataset.append({"person":person, "path": path})
    
dataset = pd.DataFrame(dataset)
dataset = dataset.groupby("person").filter(lambda x: len(x) > 10)
dataset.head(10)


# In[11]:


dataset.groupby("person").count()[:200].plot(kind='bar', figsize=(20,5))


# In[12]:


train, test = train_test_split(dataset, test_size=0.1, random_state=0)
print("Train:",len(train))
print("Test:",len(test))


# In[48]:






class Recognition(object):

    def __init__(self):

        # GRAPH
        self.graph = tf.get_default_graph()

        # Load Face Detector
        self.face_detector = MTCNN()

        # Load FaceNet
        self.facenet = FaceNet()

        # Euclidean Classifier
        self.clf = None


    def predict(self, path, threshold=None):

        if not self.clf:
            raise RuntimeError("No classifier found. Please load classifier")

        start_at = time.time()
        bounding_boxes = []
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)
        for person, confidence, box in self.__predict__(image, threshold=threshold):
            # Draw rectangle with person name
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, person, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

            bounding_boxes.append({
                "person": person,
                "confidence": confidence,
                "box": box,
            })

        # encode frame
        _, buffer = cv2.imencode('.jpg', image)

        return {
            "frame": base64.b64encode(buffer).decode('ascii'),
            "elapsed_time": (time.time() - start_at),
            "predictions": bounding_boxes

        }

    def __predict__(self, image, threshold=None):

        # Resize Image
        for encoding, face, box in self.face_encoding(image):
            # Check face size
            if (box[2] - box[0]) < MIN_FACE_SIZE[0] or                     (box[3] - box[1]) < MIN_FACE_SIZE[1]:
                yield (UNKNOWN_LABEL, 0.0, box)
            else:

                results = self.clf.predict(encoding)

                person, confidence = results["person"], results["confidence"]
                if threshold and confidence < threshold:
                    person = UNKNOWN_LABEL

                yield (person, confidence, box)

    def face_detection(self, image):

        image_to_detect = image.copy()

        # detect faces in the image
        for face_attributes in self.face_detector.detect_faces(image_to_detect):
            if face_attributes["confidence"] > FACE_CONFIDENCE:
                # extract the bounding box
                x1, y1, w, h = [max(point, 0) for point in face_attributes["box"]]
                x2, y2 = x1 + w, y1 + h

                face = image[y1:y2, x1:x2]
                # Align face
                face = Recognition.align_face(face_attributes, face.copy())

                yield (cv2.resize(face, FACE_SIZE), (x1, y1, x2, y2))

    def face_encoding(self, source_image):

        for face, box in self.face_detection(source_image):
            with self.graph.as_default():
                # Face encoding
                encoding = self.facenet.embeddings(np.expand_dims(face, axis=0))[0]

                yield (encoding, face, box)

    @staticmethod
    def align_face(face_attribute, image):
        if not face_attribute:
            return image
        # Get left and right eyes
        left_eye = face_attribute["keypoints"]["left_eye"]
        right_eye = face_attribute["keypoints"]["right_eye"]
        # Get distance between eyes
        d = math.sqrt(math.pow(right_eye[0] - left_eye[0], 2) + math.pow(right_eye[1] - left_eye[1], 2))
        a = left_eye[1] - right_eye[1]
        # get alpha degree
        alpha = (math.asin(a / d) * 180.0) / math.pi

        return imutils.rotate(image, -alpha)

    def load(self, path):

        clf = EuclideanClassifier()
        clf.load(path)

        self.clf = clf

    def save(self, path):

        self.clf.save(path)

    def fit(self, folder):

        # Initialize classifier
        #clf = EuclideanClassifier()
        clf = Cosine_Similarity()
        # Load all files
        files = []
        for ext in ALLOWED_IMAGE_TYPES:
            files.extend(glob.glob(os.path.join(folder, "*", ext), recursive=True))

        for path in tqdm.tqdm(files):
            # Load image
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get person name by folder
            person = os.path.split(os.path.split(path)[0])[1]

            # Get encoding
            for encoding, face, box in self.face_encoding(image):
                # Add to classifier
                clf.fit([encoding], [person])

        self.clf = clf

    def fit_from_dataframe(self, df, person_col="person", path_col="path"):

        # Initialize classifier
        

        clf = SVMCLASSIFIRE()
        
        #clf = EuclideanClassifier()
        #clf = Cosine_Similarity()
        for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            # Load image
            image = cv2.imread(row[path_col])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get person name by folder
            person = row[person_col]

            # Get encoding
            for encoding, face, box in self.face_encoding(image):
                # Add to classifier
                clf.fit([encoding], [person])


        self.clf = clf




# In[49]:


fr = Recognition()
fr.fit_from_dataframe(train)


# In[50]:


y_test, y_pred, y_scores = [],[],[]
for idx in range(418):
    path = test.path.iloc[idx]
    result = fr.predict(path)
    for prediction in result["predictions"]:
        y_pred.append(prediction["person"])
        y_scores.append(prediction["confidence"])
        y_test.append(test.person.iloc[idx])


# In[33]:


print(classification_report(y_test, y_pred))

