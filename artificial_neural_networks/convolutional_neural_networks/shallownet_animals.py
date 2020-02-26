from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from loaders.image_loader import ImageLoader
from preprocessors.image_preprocessor import ImageToArrayPreprocessor, ResizePreprocessor

from sha
image_to_array = ImageToArrayPreprocessor()
resizer = ResizePreprocessor(32, 32)

loader = ImageLoader(preprocessors=[resizer, image_to_array])

# loading animal dataset
print('\n[INFO] Loading data...\n')
data, labels = loader.load(path='../../datasets/cats_and_dogs')

data = data.astype('float') / 255.0

print('[INFO] splitting dataset...')
# splitting dataset:
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25, random_state=42)

print('[INFO] converting labels to VECTORS ...')
# convert labels tp vectors
lb = LabelBinarizer()
test_y = lb.fit_transform(test_y)
train_y = lb.fit_transform(train_y)

print('[INFO] COMPILING MODEL...')
model =
