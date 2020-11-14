from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

svm = SVC(kernel='linear', verbose=1)

svm.fit(X, y, )

# supports
print(svm.support_vectors_)
# indexes of supports
print(svm.support_)
print(svm.get_params())
