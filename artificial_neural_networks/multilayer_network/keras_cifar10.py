import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

input_width = 32
input_height = 32
input_depth = 3
num_classes = 10

batch_size = 32
epochs = 100
learning_rate = 0.01

# loading CIFAR-10 dataset with keras
print('LOADING DATASET...')
((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
print("DATASET LOADED")

# normalization
train_x = train_x.astype('float') / 255.0
test_x = test_x.astype('float') / 255.0

# reshaping
train_x = train_x.reshape((train_x.shape[0], input_height * input_width * input_depth))
test_x = test_x.reshape((test_x.shape[0], input_height * input_width * input_depth))

# encoding labels
lb = MultiLabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# configure model
model = Sequential()
model.add(Dense(1024, input_shape=(input_height * input_width * input_depth,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

print("TRAINING...")

sgd = SGD(learning_rate)
model.compile(optimizer=sgd, loss=['categorical_crossentropy'], metrics=['accuracy'])
model_history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size)

# evaluate model
print('EVALUATING...')

prediction = model.predict(test_x, batch_size=batch_size)

print(classification_report(test_y.argmax(1), prediction.argmax(1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), model_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), model_history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), model_history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), model_history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('CIFAR10_plot.png')
plt.show()
