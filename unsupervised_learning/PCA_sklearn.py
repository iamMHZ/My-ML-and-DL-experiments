from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Each data-point is a 8x8 image of a digit and there are 1797 samples in total
data_x, data_y = load_digits(return_X_y=True)

# apply pca
pca = PCA(n_components=2)

new_data = pca.fit_transform(data_x)

print(f'Explained variance ratio: {pca.explained_variance_ratio_}')

# plot new data
cmap = plt.get_cmap('RdBu', 10)
plt.scatter(new_data[:, 0], new_data[:, 1], c=data_y, edgecolors='black', cmap=cmap)

plt.colorbar()
plt.show()
