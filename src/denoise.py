from keras.models import load_model
from DataGenerator import DataGenerator
from skimage.measure import compare_psnr, compare_ssim
import metrics
import numpy as np
import metrics
import sys
import cv2

INPUT_SIZE = (256, 256)

def printHelp():
    print('Usage: denoise [path to model] [path to images] {path to save dir}')

def reconstructImage(batches, img_size):
    nb_cols = np.ceil(img_size[0] / INPUT_SIZE[0])
    nb_rows = np.ceil(img_size[1] / INPUT_SIZE[1])

    assert nb_cols * nb_rows == len(batches)

    # Create an empty image of maximum size that
    # includes padding to fit all batches
    size_x = int(INPUT_SIZE[0] * nb_cols)
    size_y = int(INPUT_SIZE[1] * nb_rows)
    reconstruct = np.zeros((size_y, size_x, 3), np.uint8)

    for i in range(len(batches)):
        row = np.floor(i / nb_cols)
        col = (i - (row * nb_cols))

        x_min = int(col * INPUT_SIZE[0])
        y_min = int(row * INPUT_SIZE[1])
        x_max = int((col + 1) * INPUT_SIZE[0])
        y_max = int((row + 1) * INPUT_SIZE[1])

        # Paste the batch onto the image
        batch = batches[i]
        batch = batch.reshape(INPUT_SIZE[0], INPUT_SIZE[1], 3)
        batch = batch * 255
        reconstruct[y_min:y_max, x_min:x_max] = batch[0:INPUT_SIZE[0],
                                                      0:INPUT_SIZE[1]]

    # Crop out the padding to get the original image
    original = reconstruct[0:img_size[1], 0:img_size[0]]

    return original

def reconstructImages(batches, img_size):
    nb_cols = np.ceil(img_size[0] / INPUT_SIZE[0])
    nb_rows = np.ceil(img_size[1] / INPUT_SIZE[1])
    batches_per_img = int(nb_cols * nb_rows)
    nb_images = int(len(batches) / batches_per_img)

    reconstructed = []
    for i in range(0, nb_images*batches_per_img, batches_per_img):
        img = reconstructImage(batches[i : i+batches_per_img], img_size)
        reconstructed.append(img)

    print(str(len(reconstructed)) + ' images reconstructed')

    return reconstructed

def getPredictions(model, images):
    predictions = []
    for i in range(len(images)):
        print('Processing batch ' + str(i+1) + '/' + str(len(images)))
        img = images[i]

        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        prediction = model.predict(img)
        predictions.append(prediction)

    return predictions

def saveImages(images, path):
    if path == '':
        return
    print('Saving images to ' + path + ' ...')

    for i in range(len(images)):
        cv2.imwrite(path + '/Output_' + str(i+1) + '.png', images[i])

def denoise(modelPath, imagesPath, savePath):
    datagen = DataGenerator(imagesPath, 1, INPUT_SIZE)
    batches = datagen.getAllBatchesOfX()
    model   = load_model(modelPath, custom_objects={'tf_psnr':metrics.tf_psnr,
                                                    'tf_ssim':metrics.tf_ssim})

    predictions   = getPredictions(model, batches)
    reconstructed = reconstructImages(predictions, datagen.getImageSize())

    saveImages(reconstructed, savePath)

    return reconstructed

if __name__ == '__main__':
    if len(sys.argv) < 3:
        printHelp()
        sys.exit()
    elif len(sys.argv) == 4:
        savePath = sys.argv[3]
    else:
        savePath = ''

    denoise(sys.argv[1], sys.argv[2], savePath)
