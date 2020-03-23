import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from artificial_neural_networks.convolutional_neural_networks.shallownet.shallownet import ShallowNet
from io.loaders.image_loader import ImageLoader
from preprocessors.image_preprocessors import ImageToArrayPreprocessor, ResizePreprocessor

input_width = 32
input_height = 32
input_depth = 3

num_classes = 2

batch_size = 32
epochs = 100
learning_rate = 0.005

image_to_array = ImageToArrayPreprocessor()
resizer = ResizePreprocessor(input_width, input_height)

loader = ImageLoader(preprocessors=[resizer, image_to_array])

# loading animal dataset
print('\n[INFO] Loading data...\n')

data, labels = loader.load(path='../../../datasets/cats_and_dogs')

data = data.astype('float') / 255.0

print('[INFO] splitting dataset...')

# splitting dataset:
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25, random_state=42)

print('[INFO] converting labels to VECTORS ...')
# convert labels tp vectors
lb = LabelBinarizer()
test_y = lb.fit_transform(test_y)
train_y = lb.fit_transform(train_y)

test_y = np_utils.to_categorical(test_y, num_classes=num_classes)
train_y = np_utils.to_categorical(train_y, num_classes=num_classes)

# building model
model = ShallowNet.build(input_width, input_height, input_depth, num_classes)

print('[INFO] COMPILING MODEL...')
sgd = SGD(learning_rate=learning_rate)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# training model
print('[INFO] TRAINING ...')
model_history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y), verbose=1)

# evaluating network
print('[INFO] EVALUATING network...')
predictions = model.predict(test_x, batch_size=batch_size, verbose=1)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog"]))

# plot the training loss and accuracy
print('[INFO] plotting...')
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
plt.savefig('ShallowNetAnimals.png')
plt.show()
