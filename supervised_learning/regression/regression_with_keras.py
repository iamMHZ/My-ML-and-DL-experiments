# in this module implement regression in keras
# TODO scale the data for better results
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.utils import shuffle

# load data
data_file = np.genfromtxt('../../utils/datasets/supervised dataset/fish.csv', delimiter=',')

X = data_file[1:, 5:]  # height and width of fishes
y = data_file[1:, 1]  # weight of fishes

# add a new column for the bias to the X
column = np.ones(shape=(X.shape[0], 1), dtype=np.int)
X = np.append(X, column, axis=1)
y = y.reshape(y.shape[0], 1)

# shuffle the data
X, y = shuffle(X, y)

# build the model
model = Sequential()
model.add(Dense(1, input_dim=3, activation='linear'))


model.summary()

# fit the model
learning_rate = 0.00005
epochs = 200

sgd = SGD(learning_rate=learning_rate)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

model_history = model.fit(x=X, y=y, epochs=epochs, batch_size=32)

plt.plot(np.arange(0, epochs), model_history.history['loss'], label='loss')
plt.plot(np.arange(0, epochs), model_history.history['accuracy'], label='accuracy')
plt.title('Training loss and accuracy')
plt.xlabel('Epoch')
plt.show()
