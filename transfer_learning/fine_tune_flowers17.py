from keras.applications import VGG16
from keras.models import Model, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transfer_learning.fc_head_net import FCHeadNet
from utils.loaders.image_loader import ImageLoader
from utils.preprocessors.image_preprocessors import ImageToArrayPreprocessor, AspectAwareResizePreprocessor

classes = 17
width = 224
height = 224
depth = 3

# load and split the dataset
path = 'D:\Programming\database of image\Datasets\Flower17\\train'
asa = AspectAwareResizePreprocessor(new_width=width, new_height=height)
image_to_array = ImageToArrayPreprocessor()
loader = ImageLoader(preprocessors=[image_to_array, asa])
data, labels = loader.load(path=path, )

#  applying data augmentation
generator = ImageDataGenerator(rotation_range=30, height_shift_range=0.1, width_shift_range=0.1, shear_range=0.2,
                               horizontal_flip=True)

# hot-encode labels and split dataset
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels, classes)

# split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=42, test_size=0.25)

# loading VGG16 as base model
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(width, height, depth)))

# initialize a new set of FC layers
head_model = FCHeadNet.build(base_model, classes, num_fc_nodes=256)

# place new FC layers on top of base model
model = Model(inputs=base_model.input, outputs=head_model)
