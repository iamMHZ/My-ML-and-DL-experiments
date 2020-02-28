import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from artificial_neural_networks.convolutional_neural_networks.shallownet import ShallowNet

# loading data
print('[INFO] LOADING DATA...')

((train_x, train_y), (test_x, test_y)) = cifar10.load_data()

print('[INFO] DATA LOADED SUCCESSFULLY...')

train_x = train_x.astype('float') / 255
test_x = test_x.astype('float') / 255

# initialize the label names for the CIFAR-10 dataset
label_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# convert labels to vector
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

# building model
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)

print('[INFO] COMPILING MODEL ...')
sgd = SGD(learning_rate=0.01)
model.compile(optimizer=sgd, loss=['categorical_crossentropy'], metrics=['accuracy'])

# training

print('[INFO] TRAINING...')
model_history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, epochs=40, verbose=1)

# save trained model to disk
print('[INFO] Serializing model...')
model.save('./shallowNet_weights.hdf5')

# evaluating
print("[INFO] EVALUATING...")

predictions = model.predict(test_x, batch_size=32, verbose=1)

print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog"]))

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
