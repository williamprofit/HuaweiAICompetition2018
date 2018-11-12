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
# DATA_PATH='../sample_data/'
DATA_PATH='D:/Documents/Imperial/Competitions/huaweiaicompetition/sample_data'
IMG_HEIGHT=3968
IMG_WIDTH=2976

BATCH_SIZE=1
NB_EPOCHS=1

FILTER_SIZE = (3,3)
NB_FILTERS = 64
STRIDE = 1
# Note half of total lcayers. Now can ensure total layers are always even
NB_CONV_LAYERS = 10

def createModel():
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    img_input   = Input(shape=input_shape)
    layers = [img_input]

    c0 = Conv2D(NB_FILTERS, FILTER_SIZE, strides=STRIDE, use_bias=True,
        activation="relu", input_shape=input_shape)(img_input)

    layers.append(c0) 
    # a lil dirty but ensures that the last layer will always be linked to input which is necessary.
    linking_offset = (NB_CONV_LAYERS) % 2

    for i in range(2,NB_CONV_LAYERS*2+1):
        if(i <= NB_CONV_LAYERS):
            c = Conv2D(NB_FILTERS, FILTER_SIZE, strides=STRIDE, use_bias=True,
                activation="relu")(layers[i-1])
            layers.append(c)
        else:
            relative_index = (i - NB_CONV_LAYERS)
            if(relative_index % 2 == linking_offset):
                print('linking ' + str(i) + ' with ' + str(NB_CONV_LAYERS - (relative_index)))
                
                # Manually link with input since it won't work using layers[0]
                if(NB_CONV_LAYERS - (relative_index) == 0):
                    d_pre = Conv2DTranspose(3, FILTER_SIZE,
                        strides=STRIDE, use_bias=True)(layers[i-1])
                    #ideally would want to use layers[0] instead of img_input here
                    linked_layer = add([img_input, d_pre]) 
                else:
                    d_pre = Conv2DTranspose(NB_FILTERS, FILTER_SIZE,
                        strides=STRIDE, use_bias=True)(layers[i-1])
                    linked_layer = add([layers[NB_CONV_LAYERS - (relative_index)], d_pre])

                d = Activation(activations.relu)(linked_layer) #relu after link
            else:
                d = Conv2DTranspose(NB_FILTERS, FILTER_SIZE, strides=STRIDE, use_bias=True,
                    activation="relu")(layers[i-1])

            layers.append(d)

    model = Model(inputs = img_input, outputs = layers[-1])
    model.compile(loss='mean_squared_error',
                  optimizer = Adam(lr=0.0001),
                  metrics=['accuracy'])
    return model
    
def main():
    generator = DataGenerator(DATA_PATH, BATCH_SIZE)

    model = createModel()
    #model.fit_generator(generator, epochs=NB_EPOCHS)

if __name__ == '__main__':
    main()


# Residual connection on a convolution layer

# For more information about residual networks, see Deep Residual Learning for Image Recognition.

# from keras.layers import Conv2D, Input

# # input tensor for a 3-channel 256x256 image
# x = Input(shape=(256, 256, 3))
# # 3x3 conv with 3 output channels (same as input channels)
# y = Conv2D(3, (3, 3), padding='same')(x)
# # this returns x + y.
# z = keras.layers.add([x, y])
