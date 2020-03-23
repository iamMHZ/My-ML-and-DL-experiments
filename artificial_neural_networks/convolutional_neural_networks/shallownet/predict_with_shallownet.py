from io.loaders import model_loader
from io.loaders.image_loader import ImageLoader
from preprocessors.image_preprocessors import ResizePreprocessor, ImageToArrayPreprocessor

'''
    1- loading shallowNet trained model from disk
    2- predicting a set of randomly choose images with it 
    3- showing prediction result
'''

# loading and preprocessing images in the exact same way that we trained shallowNet

image_to_array = ImageToArrayPreprocessor()
resizer = ResizePreprocessor(32, 32)

data_path = ''
data_loader = ImageLoader(preprocessors=[resizer, image_to_array])
data, labels = data_loader.load(data_path)  # load with cv2 module

data = data.astype('float') / 255.0

# path to pre-trained model
model_path = ''
# load model
model = model_loader.load(model_path)

# predict data with model
predictions = model.predict(data, batch_size=32).argmax(axis=1)

# print class predicted class labels
class_labels = ['cat', 'dog']
for pre in predictions:
    print(pre)
