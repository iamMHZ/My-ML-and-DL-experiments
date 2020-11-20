import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs


class KMeans:
    # TODO add cost of the clustering

    # Initialization method for centers
    INITIALIZE_RANDOMLY = 0
    INITIALIZE_FROM_DATA = 1
    # End criteria
    # TODO add another criteria
    MAX_ITERATION_END_CRITERIA = 0
    NO_CHANGE_IN_CENTERS_END_CRITERIA = 1

    def __init__(self, data, K, initialization_method=INITIALIZE_FROM_DATA):

        self.num_centers = K
        self.data = data

        # initialize centers
        if initialization_method == KMeans.INITIALIZE_RANDOMLY:
            # randomly initialize centers
            # this method may cause some centers to have no data
            self.centers = np.random.random((self.num_centers, *data.shape[1:]))
        else:
            random_indexes = np.random.randint(low=0, high=self.data.shape[0], size=self.num_centers)
            # choose centers from data randomly
            self.centers = self.data[random_indexes]

        # initialize an array as the center of each data point; initially all of them are in one cluster
        self.data_center_ids = np.zeros(data.shape[0])

    def start_clustering(self, end_criteria=NO_CHANGE_IN_CENTERS_END_CRITERIA, max_iterations=300):

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

            # check the end criteria
            if num_iterations >= max_iterations and end_criteria == KMeans.MAX_ITERATION_END_CRITERIA:
                print('MAX ITERATION END CRITERIA ===> FINISHED')
                break
            elif np.allclose(self.data_center_ids,
                             old_data_center_ids) and end_criteria == KMeans.NO_CHANGE_IN_CENTERS_END_CRITERIA:
                print('NO CHANGES IN CENTERS END CRITERIA ===> FINISHED ')
                break

            # update old centers for the next iteration
            old_data_center_ids = self.data_center_ids.copy()
            num_iterations += 1

    def plot(self):
        # for now, in case of 2D data plot data

        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.data_center_ids)

        # TODO debug the centers plot colors
        plt.scatter(self.centers[:, 0], self.centers[:, 1], marker='*', s=200, c=np.arange(0, self.centers.shape[0]),
                    edgecolors='black', zorder=1)
        plt.show(delay=100)


if __name__ == '__main__':
    data, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1)

    # for 2D data
    plt.scatter(data[:, 0], data[:, 1])

    plt.show()

    km = KMeans(data=data, K=5, initialization_method=KMeans.INITIALIZE_FROM_DATA)

    km.start_clustering(end_criteria=KMeans.MAX_ITERATION_END_CRITERIA)
