from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.optimizers import Adam, SGD
from keras.models import Model, Sequential

from DataGenerator import DataGenerator


DATA_PATH='../sample_data/'
IMG_HEIGHT=3968
IMG_WIDTH=2976

BATCH_SIZE=1
NB_EPOCHS=250


def createModel():
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2DTranspose(64, (3,3), activation='relu'))
    model.add(Conv2DTranspose(3, (3,3), activation='relu'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def main():
    generator = DataGenerator(DATA_PATH, BATCH_SIZE)

    model = createModel()
    model.fit_generator(generator, epochs=2)

if __name__ == '__main__':
    main()
