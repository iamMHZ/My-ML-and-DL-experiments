from keras.applications import VGG16
from keras.models import Model, Input
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from computer_vision.transfer_learning.fc_head_net import FCHeadNet
from utils.loaders.image_loader import ImageLoader
from utils.model_monitoring.training_monitoring import TrainingMonitor
from utils.preprocessors.image_preprocessors import ImageToArrayPreprocessor, AspectAwareResizePreprocessor

classes = 17
width = 224
height = 224
depth = 3

batch_size = 32

# load and split the dataset
path = 'D:\Programming\database of image\Datasets\Flower17\\train'
asa = AspectAwareResizePreprocessor(new_width=width, new_height=height)
image_to_array = ImageToArrayPreprocessor()
loader = ImageLoader(preprocessors=[image_to_array, asa])
data, labels = loader.load(path=path, )

#  applying data augmentation
generator = ImageDataGenerator(rotation_range=30, height_shift_range=0.1, width_shift_range=0.1, shear_range=0.2,
                               horizontal_flip=True)
callbacks = [TrainingMonitor('VGG16FineTube')]

# hot-encode labels and split dataset
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels, classes)

# split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=42, test_size=0.25)

# loading VGG16 as base model
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(width, height, depth)))
print('VGG16 loaded')

# initialize a new set of FC layers
head_model = FCHeadNet.build(base_model, classes, num_fc_nodes=256)

# place new FC layers on top of base model
model = Model(inputs=base_model.input, outputs=head_model)

# First of all we need to warm up new FC layers so we freeze base model's layers and
# then we start training new FC layers

print('Warming up new FC layers')
for layer in base_model.layers:
    layer.trainable = False

# warm up FC layers
rms_prop = RMSprop(learning_rate=0.001)
model.compile(optimizer=rms_prop, loss="categorical_crossentropy", metrics=['accuracy'])
model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) // batch_size,
                    epochs=25,
                    validation_data=(x_test, y_test), callbacks=callbacks)

predictions = model.predict(x_test, batch_size=batch_size, verbose=1)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[n for n in label_encoder.classes_]))

# now that FC layers have been warmed up
# lets train last set of CONV layers
print('Training CONV layers')
for layer in base_model.layers[15:]:
    layer.trainable = True

sgd = SGD(learning_rate=0.001)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) // batch_size,
                    epochs=100,
                    validation_data=(x_test, y_test), callbacks=callbacks)

predictions = model.predict(x_test, batch_size=batch_size, verbose=1)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[n for n in label_encoder.classes_]))

print('Saving model')
model.save('./FineTunedVGG16OnFlowers17.hdf5')
