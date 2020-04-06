from keras.layers.core import Flatten, Dense, Dropout


class FCHeadNet:

    @staticmethod
    def build(base_model, classes, num_fc_nodes):
        # read https://keras.io/getting-started/functional-api-guide/

        # head_model will be placed at the top of the based model
        head_model = base_model.output

        head_model = Flatten(name='flatten')(head_model)
        head_model = Dense(num_fc_nodes, activation='relu')(head_model)
        head_model = Dropout(0.5)(head_model)

        head_model = Dense(classes, activation='softmax')(head_model)

        return head_model
