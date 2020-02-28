import keras.backend as keras_backed
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from artificial_neural_networks.convolutional_neural_networks.lenet.lenet import LeNet

input_width = 28
input_height = 28
input_depth = 1
num_classes = 10

batch_size = 128
epochs = 20

# load mnist dataset

print('[INFO] LOADING DATA...')

dataset = datasets.fetch_openml('mnist_784')
data = dataset.data / 255.0

# input type of LeNet is 28*28*1 so we need to reshape the data
data = data.reshape(data.shape[0], input_height, input_width, input_depth)
if keras_backed.image_data_format() == 'channels_first':
    data = data.reshape(data.shape[0], input_depth, input_height, input_width)

# split dataset :
train_x, test_x, train_y, test_y = train_test_split(data, dataset.target.astype('int'), test_size=0.25,
                                                    random_state=42)

# convert labels to vectors
lb = LabelBinarizer()
test_y = lb.fit_transform(test_y)
train_y = lb.fit_transform(train_y)

print('[INFO] DATA LOADED')

# build model
model = LeNet.build(input_width, input_height, input_depth, num_classes)

print('[INFO] COMPILING MODEL...')

sgd = SGD(learning_rate=0.01)
model.compile(optimizer=sgd, loss=['categorical_crossentropy'], metrics=['accuracy'])

# training
print('[INFO] TRAINING...')

model_history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1,
                          validation_data=(test_x, test_y))

# evaluating network
print('[INFO] EVALUATING network...')
predictions = model.predict(test_x, batch_size=batch_size, verbose=1)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(l) for l in lb.classes_]))

# plot the training loss and accuracy
print('[INFO] plotting...')
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), model_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), model_history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), model_history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), model_history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('ShallowNetAnimals.png')
plt.show()
