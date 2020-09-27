# in this module I learn the ropes of the regression implementation using the sklearn
# TODO scale the data for better results
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle

# load data

# https://www.kaggle.com/aungpyaeap/fish-market?select=Fish.csv
data_file = np.genfromtxt('../utils/datasets/supervised dataset/fish.csv', delimiter=',')

X = data_file[1:, 5:]  # height and width of fishes
y = data_file[1:, 1]  # weight of fishes

# add a new column for the bias to the X

column = np.ones(shape=(X.shape[0], 1), dtype=np.int)
X = np.append(X, column, axis=1)

y = y.reshape(y.shape[0], 1)

# shuffle the data

X, y = shuffle(X, y)

# use sklearn and apply regression

linear_regression = LinearRegression()

# train
linear_regression.fit(X[:150, :], y[:150])
score = linear_regression.score(X[:150, :], y[:150])

print(score)

print(linear_regression.coef_)

print(linear_regression.get_params())

# test

prediction = linear_regression.predict(X[150:, :])

print(prediction)
