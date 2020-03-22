import h5py


class HDF5DatasetWriter:

    def __int__(self, output_path, height, width, dataset_name_keyword='images', buffer_size=1000):
        self.dataset_dimension = (height, width)
        self.dataset_name_keyword = dataset_name_keyword
        self.buffer_size = buffer_size

        self.hdf5_file = h5py.File(output_path, mode='w')
        self.data_storage = self.hdf5_file.create_dataset(name=dataset_name_keyword, shape=self.dataset_dimension,
                                                          dtype='float')
        self.labels_storage = self.hdf5_file.create_dataset(name='labels', shape=(height,), dtype='int')

        self.buffer_size = buffer_size
        self.buffer = {'data': [], 'labels': []}

        self.current_data_index = 0

    def add(self, data_row, associated_labels):
        self.buffer['data'].extend(data_row)
        self.buffer['labels'].extend(associated_labels)

        if len(self.buffer['data']) >= self.buffer_size:
            self.flush()

    def flush(self):
        end_index = self.current_data_index + len(self.buffer['data'])

        self.data_storage[self.current_data_index:end_index] = self.buffer['data']
        self.labels_storage[self.current_data_index:end_index] = self.buffer['labels']

        self.current_data_index = end_index
        self.buffer['data'] = []
        self.buffer['labels'] = []

    def close(self):

        if len(self.buffer['data']) > 0:
            self.flush()

        self.hdf5_file.close()

    def store_class_labels(self, class_labels):
        # create a dataset to store the actual class label names,
        # then store the class labels
        dt = h5py.string_dtype(encoding='utf-8')
        label_set = self.hdf5_file.create_dataset("label_names",
                                                  (len(class_labels),), dtype=dt)
        label_set[:] = class_labels
