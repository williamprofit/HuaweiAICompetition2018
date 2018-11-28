from keras.models import load_model
from DataGenerator import DataGenerator
from skimage.measure import compare_psnr, compare_ssim
import shutil
import metrics
import numpy as np
import metrics
import sys
import cv2
import os

INPUT_SIZE = (256, 256)
PADDING = 32
PERMUTATIONS=False
#Keep linewidth even
LINEWIDTH = 20

def printHelp():
    print('Usage: denoise [path to model] [path to images] {path to save dir}')

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

def addPadding(originalImgs, padding):
    bigImgs = []
    for img in originalImgs:
        size_x = img.shape[1]
        size_y = img.shape[0]
        big_size_x = size_x + 2 * padding
        big_size_y = size_y + 2 * padding
        bigImg = np.zeros((big_size_y, big_size_x, 3), np.float)

        # Copy the original image on the center
        bigImg[padding:size_y+padding, padding:size_x+padding] = img

        # Create mirrored borders
        # Top
        bigImg[0:padding, padding:size_x+padding] = cv2.flip(img[0:padding, :], 0)
        # Bot
        bigImg[size_y+padding:big_size_y, padding:size_x+padding] = cv2.flip(img[size_y-padding:size_y, :], 0)
        # Left
        bigImg[padding:size_y+padding, 0:padding] = cv2.flip(img[:, 0:padding], 1)
        # Right
        bigImg[padding:size_y+padding, size_x+padding:big_size_x] = cv2.flip(img[:, size_x-padding:size_x], 1)

        # Create mirrored corners
        # Top Left
        bigImg[0:padding, 0:padding] = cv2.flip(bigImg[0:padding, padding:2*padding], 1)
        # Top Right
        bigImg[0:padding, size_x+padding:big_size_x] = cv2.flip(bigImg[0:padding, size_x:size_x+padding], 1)
        # Bot Left
        bigImg[size_y+padding:big_size_y, 0:padding] = cv2.flip(bigImg[size_y:size_y+padding, 0:padding], 0)
        # Bot Right
        bigImg[size_y+padding:big_size_y, size_x+padding:big_size_x] = cv2.flip(bigImg[size_y:size_y+padding, size_x:size_x+padding], 0)

        bigImgs.append(bigImg)

    return bigImgs

def removeNoisyGrid(smallImages, bigImages, padding):
    xSize = smallImages[0].shape[1]
    ySize = smallImages[0].shape[0]

        # for simplicity, trim big image into same size as small image
    for i in range(len(bigImages)):
        bigImages[i] = bigImages[i][padding:ySize+padding, padding:xSize+padding]

    halfline = int(LINEWIDTH/2)

    for i in range(len(smallImages)):
        # Clean vertical lines
        # Deal with border first
        smallImages[i][:, 0:halfline] = bigImages[i][:, 0:halfline]
        smallImages[i][:, xSize-halfline:] = bigImages[i][:, xSize-halfline:]
        # Deal with inner grid lines
        for x in range(INPUT_SIZE[0]-halfline, xSize-LINEWIDTH, INPUT_SIZE[0]):
            smallImages[i][:, x:x+LINEWIDTH] = bigImages[i][:, x:x+LINEWIDTH]

        # Clean horizontal lines
        # Deal with border first
        smallImages[i][0:halfline, :] = bigImages[i][0:halfline, :]
        smallImages[i][ySize-halfline:, :] = bigImages[i][ySize-halfline:, :]
        # Deal with inner grid lines
        for y in range(INPUT_SIZE[1]-halfline, ySize-LINEWIDTH, INPUT_SIZE[1]):
            smallImages[i][y:y+LINEWIDTH, :] = bigImages[i][y:y+LINEWIDTH, :]

    return smallImages

def removeNoisyEdgeDots(smallImages, bigImages, padding, bigPadding):
    xSize = smallImages[0].shape[1]
    ySize = smallImages[0].shape[0]
    halfline = int(LINEWIDTH/2)
    xOverflow = xSize % INPUT_SIZE[0]
    for i in range(len(bigImages)):
        bigImages[i] = bigImages[i][bigPadding:ySize+bigPadding, bigPadding:xSize+bigPadding]

    for i in range(len(smallImages)):
# Along x
        x = xSize-(INPUT_SIZE[0]-padding)-halfline
        smallImages[i][0:halfline, x:x+LINEWIDTH] = bigImages[i][0:halfline, x:x+LINEWIDTH]
        for x in range(INPUT_SIZE[0]-padding-halfline, xSize-LINEWIDTH, INPUT_SIZE[0]):
            smallImages[i][0:halfline, x:x+LINEWIDTH] = bigImages[i][0:halfline, x:x+LINEWIDTH]

        for y in range(INPUT_SIZE[1]-halfline, ySize-LINEWIDTH, INPUT_SIZE[1]):
            x = xSize-(INPUT_SIZE[0]-padding)-halfline
            smallImages[i][y:y+LINEWIDTH, x:x+LINEWIDTH] = bigImages[i][y:y+LINEWIDTH, x:x+LINEWIDTH]
            for x in range(INPUT_SIZE[0]-padding-halfline, xSize-LINEWIDTH, INPUT_SIZE[0]):
                smallImages[i][y:y+LINEWIDTH, x:x+LINEWIDTH] = bigImages[i][y:y+LINEWIDTH, x:x+LINEWIDTH]

        x = xSize-(INPUT_SIZE[0]-padding)-halfline
        smallImages[i][ySize-halfline:, x:x+LINEWIDTH] = bigImages[i][ySize-halfline:, x:x+LINEWIDTH]
        for x in range(INPUT_SIZE[0]-padding-halfline, xSize-LINEWIDTH, INPUT_SIZE[0]):
            smallImages[i][ySize-halfline:, x:x+LINEWIDTH] = bigImages[i][ySize-halfline:, x:x+LINEWIDTH]

# Along y
        y = ySize-(INPUT_SIZE[1]-padding)-halfline
        smallImages[i][y:y+LINEWIDTH, 0:halfline] = bigImages[i][y:y+LINEWIDTH, 0:halfline]
        for y in range(INPUT_SIZE[1]-padding-halfline, ySize-LINEWIDTH, INPUT_SIZE[1]):
            smallImages[i][y:y+LINEWIDTH, 0:halfline] = bigImages[i][y:y+LINEWIDTH, 0:halfline]

        for x in range(INPUT_SIZE[0]-halfline, xSize-LINEWIDTH, INPUT_SIZE[0]):
            y = ySize-(INPUT_SIZE[1]-padding)-halfline
            smallImages[i][y:y+LINEWIDTH, x:x+LINEWIDTH] = bigImages[i][y:y+LINEWIDTH, x:x+LINEWIDTH]
            for y in range(INPUT_SIZE[1]-padding-halfline, ySize-LINEWIDTH, INPUT_SIZE[1]):
                smallImages[i][y:y+LINEWIDTH, x:x+LINEWIDTH] = bigImages[i][y:y+LINEWIDTH, x:x+LINEWIDTH]

        y = ySize-(INPUT_SIZE[1]-padding)-halfline
        smallImages[i][y:y+LINEWIDTH:, xSize-halfline:] = bigImages[i][y:y+LINEWIDTH:, xSize-halfline:]
        for y in range(INPUT_SIZE[1]-padding-halfline, ySize-LINEWIDTH, INPUT_SIZE[1]):
            smallImages[i][y:y+LINEWIDTH:, xSize-halfline:] = bigImages[i][y:y+LINEWIDTH:, xSize-halfline:]

    return smallImages


def createBigImages(model, originalImages, padding):
    tempDirectory = "temp_bigimages"
    if os.path.exists(tempDirectory):
        shutil.rmtree(tempDirectory)
    os.makedirs(tempDirectory)

    bigImages = addPadding(originalImages, padding)
    saveImages(bigImages, tempDirectory)

    bigDatagen = DataGenerator(tempDirectory, 1, INPUT_SIZE, permutations=PERMUTATIONS)
    bigBatches     = bigDatagen.getAllBatchesOfX()
    bigPredictions = getPredictions(model, bigBatches)

    bigReconstructed = bigDatagen.reconstructImages(bigPredictions)
    saveImages(bigImages, tempDirectory)

    return bigReconstructed

def denoise(modelPath, imagesPath, savePath):
    usePermutations = False
    datagen = DataGenerator(imagesPath, 1, INPUT_SIZE, permutations=PERMUTATIONS)

    batches = datagen.getAllBatchesOfX()
    if(not modelPath.endswith(".hdf5")):
        modelPath += ".hdf5"

    model   = load_model(modelPath, custom_objects={'tf_psnr':metrics.tf_psnr,
                                                    'tf_ssim':metrics.tf_ssim})
    predictions   = getPredictions(model, batches)
    smallReconstructed = datagen.reconstructImages(predictions)

    bigImages = createBigImages(model, datagen.getAllX(), PADDING)
    biggerImages = createBigImages(model, datagen.getAllX(), PADDING + 3 * LINEWIDTH)

    reconstructed = removeNoisyGrid(smallReconstructed, bigImages, PADDING)
    reconstructed = removeNoisyEdgeDots(reconstructed, biggerImages, PADDING, PADDING + 3 * LINEWIDTH)
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
