from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, ReLU
from tensorflow.keras import initializers

def FSRCNN(d: int, s: int, m: int, input_size: tuple, upscaling_factor: int, color_channels: int):
    """
    FSRCNN model implementation from https://arxiv.org/abs/1608.00367
    
    Sigmoid Activation in the output layer is not in the original paper.
    But it is needed to align model prediction with ground truth HR images
    so their values are in the same range [0,1].
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(input_size[0], input_size[1], color_channels)))

    # feature extraction
    model.add(Conv2D(kernel_size=5,filters=d,padding="same"))
    model.add(ReLU())

    # shrinking
    model.add(Conv2D(kernel_size=1, filters=s, padding="same"))
    model.add(ReLU())

    # mapping
    for _ in range(m):
        model.add(Conv2D(kernel_size=3,filters=s,padding="same"))
        model.add(ReLU())

    # expanding
    model.add(Conv2D(kernel_size=1, filters=d, padding="same"))
    model.add(ReLU())

    # deconvolution
    model.add(Conv2DTranspose(
            kernel_size=9,
            filters=color_channels,
            strides=(2,2),
            padding="same",
            kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.001)))

    return model
