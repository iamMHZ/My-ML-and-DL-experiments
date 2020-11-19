import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs


class KMeans:
    # TODO debug the centers plot colors
    # TODO  add end criteria as an enum or sth
    # TODO change initialization

    def __init__(self, data, K):
        self.num_centers = K
        self.data = data
        # initialize centers
        self.centers = np.random.random((self.num_centers, *data.shape[1:]))

        # initialize center of each data point; initially all data is belonged to one cluster
        self.data_center_ids = np.zeros(data.shape[0])

    def start_clustering(self, max_iterations=300):

        old_data_center_ids = self.data_center_ids.copy()  # don't forget .copy()

        num_iterations = 0
        while True:

            print(f'##Iteration number {num_iterations}')

            # update centers of each data
            for i, data_point in enumerate(self.data):
                center_id_i = np.argmin(np.linalg.norm(data_point - self.centers, axis=1))
                self.data_center_ids[i] = center_id_i

            # update centers
            for i, center in enumerate(self.centers):
                indexes = np.argwhere(self.data_center_ids == i)
                data_clusters = self.data[indexes.reshape(indexes.shape[0])]
                # check that this cluster has assigned data to it
                if len(data_clusters) > 0:
                    mean = np.mean(data_clusters, axis=0)
                    self.centers[i] = mean

            print('Plotting... ')
            self.plot()

            if num_iterations >= max_iterations or np.allclose(self.data_center_ids, old_data_center_ids):
                print('Meet a criteria')
                break

            old_data_center_ids = self.data_center_ids.copy()
            num_iterations += 1

    def plot(self):
        # for now, in case of 2D data plot data

        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.data_center_ids)
        plt.scatter(self.centers[:, 0], self.centers[:, 1], marker='*', s=200, c=np.arange(0, self.centers.shape[0]),
                    edgecolors='black', zorder=1)
        plt.show(delay=100)


if __name__ == '__main__':
    data, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1)

    # for 2D data
    plt.scatter(data[:, 0], data[:, 1])

    plt.show()

    km = KMeans(data=data, K=4)

    km.start_clustering()
