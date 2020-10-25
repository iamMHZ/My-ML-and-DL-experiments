"""
A simple the one versus rest logistic regression classifier is implemented in this module

"""
import matplotlib.pyplot as  plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from supervised_learning.classification.logistic_regression.logistic_regression import fit

NUM_CLASSES = 3
LEARNING_RATE = 0.0001
EPOCHS = 300

NUM_FEATURES = 2

# make dummy data with sklearn and plot them
X, y = make_blobs(n_samples=5000, n_features=NUM_FEATURES, centers=NUM_CLASSES, random_state=42)
color = ['red', 'green', 'blue']
plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.scatter(X, y, c=y)
plt.show()

# add a column for the bias (bias trick) ==> everything is vectorized
ones_column = np.ones((X.shape[0], 1), np.float)
X = np.append(X, ones_column, axis=1)

y = y.reshape(y.shape[0], 1)

# split the data

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=18)

weights = np.zeros((NUM_FEATURES + 1, NUM_CLASSES))  # +1 is the bias
# weights=[]

# train  classifiers
for i in range(NUM_CLASSES):
    print(f'\nTraining classifier {i} \n')

    # change labels to 0 and 1 for each classifier
    changed_labels = train_y.copy()
    changed_labels[changed_labels == i] = 0
    changed_labels[changed_labels != i] = 1

    # train the ith classifier
    weight_i = fit(train_x, changed_labels, learning_rate=LEARNING_RATE, epochs=EPOCHS)

    # append weights
    weights[:, i] = weight_i[:, 0]
    # weights.append(weight_i)

print('Weights: ')
# weights = np.array(weights)
print(weights)

# prediction
print('Predictions: ')
predictions = test_x @ weights

print(predictions)
report=classification_report(test_y, predictions.argmax(axis=1))
print(report)
