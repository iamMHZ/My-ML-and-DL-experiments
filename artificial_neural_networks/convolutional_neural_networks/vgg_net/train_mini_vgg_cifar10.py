import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from artificial_neural_networks.convolutional_neural_networks.vgg_net.mini_vgg_net import MiniVGGNet

input_width = 32
input_height = 32
input_depth = 3

num_classes = 10

epochs = 40
batch_size = 64

learning_rate = 0.01
momentum_term = 0.9
decay = learning_rate / epochs

# loading data
print('[INFO] LOADING DATA...')

((train_x, train_y), (test_x, test_y)) = cifar10.load_data()

print('[INFO] DATA LOADED SUCCESSFULLY')

train_x = train_x.astype('float') / 255.0
test_x = test_x.astype('float') / 255.0

# initialize the label names for the CIFAR-10 dataset
label_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# convert label to vectors
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

# build models
model = MiniVGGNet.build(input_width, input_height, input_depth, num_classes, batch_normalization=True)

print('[INFO] Compiling model...')
sgd = SGD(learning_rate, momentum=momentum_term, decay=decay, nesterov=True)
model.compile(optimizer=sgd, loss=['categorical_crossentropy'], metrics=['accuracy'])

# training
print('[INFO] Training...')
model_history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1,
                          validation_data=(test_x, test_y))

# save trained model to disk
print('[INFO] Serializing model...')
model.save('./MiniVGG.hdf5')

# evaluating
print("[INFO] EVALUATING...")

predictions = model.predict(test_x, batch_size=batch_size, verbose=1)

print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

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
plt.savefig('MiniVGG.png')
plt.show()
