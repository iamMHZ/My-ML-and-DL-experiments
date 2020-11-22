import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs


class KMeans:
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

        self.clustering_cost = None

    def start_clustering(self, max_iterations=300, end_criteria=NO_CHANGE_IN_CENTERS_END_CRITERIA):

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

    def cost(self):
        # mean squared error of the clustering ( or Euclidean distance or how far is data for its center)
        # this can be used to determine the right K for the algorithm also( Elbow technique)

        self.clustering_cost = 0

        for i in range(self.num_centers):
            indexes = np.argwhere(self.data_center_ids == i)
            data_clusters = self.data[indexes.reshape(indexes.shape[0])]
            if len(data_clusters) > 0:
                dist = np.linalg.norm(data_clusters - self.centers[i], axis=1)
                self.clustering_cost += np.sum(dist)

        self.clustering_cost /= self.data.shape[0]

        print(f'Print clustering cost with these number of centers(K) is {self.clustering_cost}')

    def plot(self):
        # for now, in case of 2D data, plot data
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.data_center_ids)

        # TODO debug the centers  colors
        plt.scatter(self.centers[:, 0], self.centers[:, 1], marker='*', s=200, c=np.arange(0, self.centers.shape[0]),
                    edgecolors='black', zorder=1)
        plt.show()


if __name__ == '__main__':
    data, _ = make_blobs(n_samples=400, centers=3, n_features=2, cluster_std=1, random_state=33)

    # for 2D data
    plt.scatter(data[:, 0], data[:, 1])

    plt.show()

    km = KMeans(data=data, K=5, initialization_method=KMeans.INITIALIZE_FROM_DATA)

    km.start_clustering(end_criteria=KMeans.NO_CHANGE_IN_CENTERS_END_CRITERIA)
    km.cost()
