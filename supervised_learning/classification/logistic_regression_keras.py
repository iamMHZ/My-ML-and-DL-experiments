"""
Logistic regression with keras
"""

import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD

from matplotlib import pyplot as plt

LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 4

# https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival
data_file = np.genfromtxt('../../utils/datasets/supervised dataset/haberman.txt', delimiter=',')

X = data_file[:, :2]
y = data_file[:, 3]

#  labels are 1 (survived) and 2 (died)
# change 2 to 0
y[y == 2] = 0

ones_column = np.ones((X.shape[0], 1), np.float)
X = np.append(X, ones_column, axis=1)
y = y.reshape(y.shape[0], 1)

# build a model
input_layer = Input(shape=(X.shape[1],))
output_layer = Dense(units=1, activation='sigmoid')(input_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()
model.compile(optimizer=SGD(learning_rate=LEARNING_RATE), loss='binary_crossentropy')

his = model.fit(x=X, y=y, epochs=EPOCHS, batch_size=BATCH_SIZE)

plt.plot(np.arange(0, EPOCHS), his.history['loss'], label='loss')
plt.legend()
plt.ylabel('Epoch')
plt.ylabel('Metric')
plt.show()
