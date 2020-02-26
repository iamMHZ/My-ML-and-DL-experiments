from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer

print("Loading MNIST ...")
dataset = datasets.fetch_openml('mnist_784')
print("MNIST loaded ")

# apply normalization
data = dataset.data.astype('float') / 255.0
# data = dataset.data.astype('float')
# data = (data - data.min()) / (data.max() - data.min())

# split the dataset
(train_x, test_x, train_y, test_y) = train_test_split(data, dataset.target, test_size=0.25)

# convert labels to vectors
lb = MultiLabelBinarizer()
# lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)


print("Training ...")
# define network architecture:
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))


# train model using SGD
sgd = SGD(0.01)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model_history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, batch_size=128)

print("Evaluating Network : ")

prediction = model.predict(test_x, batch_size=128)

print(
    classification_report(test_y.argmax(axis=1), prediction.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
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
plt.savefig('MNIST_plot.png')
plt.show()
