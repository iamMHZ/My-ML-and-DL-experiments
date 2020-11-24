"""
Kmeams on the 8x8 digit dataset using sklearn
"""
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits


def plot(data, title):
    plt.imshow(data.reshape((8, 8)), cmap='gray')
    plt.title(title)
    plt.show()


# Each data-point is a 8x8 image of a digit and there are 1797 samples in total
data_x, data_y = load_digits(return_X_y=True)

print(data_x.shape)
print(data_y.shape)

# apply Kmeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(data_x)

# print the center that each data belongs to
print(kmeans.labels_)
# print centers
centers = kmeans.cluster_centers_
print(centers)
print(len(centers))
# plot centers
for i, center in enumerate(centers):
    plot(center, title=str(i))
