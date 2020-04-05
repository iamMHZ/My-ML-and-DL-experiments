from keras.applications import VGG16


def print_layer_names(model):
    for i, layer in enumerate(model.layers):
        print(f'{i}- {layer.__class__.__name__}')


if __name__ == '__main__':
    model = VGG16(weights='imagenet', include_top=False)

    print_layer_names(model)
