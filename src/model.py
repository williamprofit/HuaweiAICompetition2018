from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.optimizers import Adam, SGD
from keras.models import Model, Sequential
from keras import activations
from DataGenerator import DataGenerator
import metrics
from keras.backend import equal

# Hyperparameters:
DATA_PATH = 'sample_data/'
INPUT_SIZE = (256, 256)
IMG_SIZE = (2976, 3968)

BATCH_SIZE = 16
NB_EPOCHS = 1

FILTER_SIZE = (3,3)
NB_FILTERS = 64
STRIDE = 1
USE_BIAS = True

# Note half of total layers to ensure total layers are always even
NB_CONV_LAYERS = 10

def createModel():
    # (height, width, channels)
    input_shape = (INPUT_SIZE[1], INPUT_SIZE[0], 3)
    img_input   = Input(shape=input_shape)

    layers = [img_input]

    c0 = Conv2D(NB_FILTERS, FILTER_SIZE, strides=STRIDE, use_bias=True,
        activation="relu", input_shape=input_shape)(img_input)
    layers.append(c0)

    for i in range(2,NB_CONV_LAYERS*2):
        if(i <= NB_CONV_LAYERS):
            addConvLayer(layers)
        else:
            relative_index = (i - NB_CONV_LAYERS)
            if(relative_index % 2 == 1):
                addSkipConnection(layers,
                                  layers[NB_CONV_LAYERS - (relative_index)])
            else:
                addDeconvLayer(layers)

    # Final image has 3 channels, hence 3 filters
    addSkipConnection(layers, layers[0], nb_filter=3)

    model = Model(inputs = img_input, outputs = layers[-1])
    model.compile(loss='mean_squared_error',
                  optimizer = Adam(lr=0.0001),
                  metrics=['accuracy'])
    print(model.summary())

    return model

def addConvLayer(layers, stride=STRIDE):
    c = Conv2D(NB_FILTERS, FILTER_SIZE, strides=stride, use_bias=USE_BIAS,
                activation="relu")(layers[-1])
    layers.append(c)

def addDeconvLayer(layers, stride=STRIDE):
    d = Conv2DTranspose(NB_FILTERS, FILTER_SIZE, strides=STRIDE, use_bias=True,
                    activation="relu")(layers[-1])
    layers.append(d)

def addSkipConnection(layers, skipped_layer, nb_filter=NB_FILTERS, filter_size=FILTER_SIZE):
    deconv = Conv2DTranspose(nb_filter, filter_size,
                    strides=STRIDE, use_bias=USE_BIAS)(layers[-1])
    linked_layer = add([skipped_layer, deconv])
    activated_layer = Activation(activations.relu)(linked_layer)

    layers.append(activated_layer)

def main():
    generator = DataGenerator(DATA_PATH, BATCH_SIZE, IMG_SIZE, INPUT_SIZE)

    model = createModel()
    model.fit_generator(generator, epochs=NB_EPOCHS)

    model.save('../models/model')

if __name__ == '__main__':
    main()
