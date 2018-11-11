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


# Hyperparameters:
DATA_PATH='../sample_data/'
IMG_HEIGHT=3968
IMG_WIDTH=2976

BATCH_SIZE=1
NB_EPOCHS=250

FILTER_SIZE = (3,3)
NO_FILTERS = 64
STRIDE = 3
# Note half of total layers. Now can ensure total layers are always even
NO_CONV_LAYERS = 15

def createModel():
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    model = Sequential()
    c0 = Conv2D(NO_FILTERS, FILTER_SIZE, STRIDE,use_bias=True, 
        activation="relu", input_shape=input_shape) 
    model.add(c0)

    for i in range(1,NO_CONV_LAYERS*2):
        if(i < NO_CONV_LAYERS):
            c = Conv2D(NO_FILTERS, FILTER_SIZE, STRIDE,use_bias=True, 
                activation="relu")(model.layers[i-1])
            model.add(c)
        else:
            relative_index = (i - NO_CONV_LAYERS)
            if(relative_index % 2 == 0):
                d_pre = Conv2DTranspose(NO_FILTERS, FILTER_SIZE, 
                    STRIDE, use_bias=True)(model.layers[i-1]) #note no relu here
                d = activations.relu( add([model.inputs[relative_index], d_pre])) #relu here
            else:
                d = Conv2DTranspose(NO_FILTERS, FILTER_SIZE, STRIDE, use_bias=True, 
                    activation="relu")(model.layers[i-1])
            
            model.add(d)

    model.compile(loss='mean_squared_error',
                  optimizer = Adam(lr=0.0001),
                  metrics=['accuracy', metrics.psnr])

    return model

def main():
    generator = DataGenerator(DATA_PATH, BATCH_SIZE)

    model = createModel()
    model.fit_generator(generator, epochs=2)

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
