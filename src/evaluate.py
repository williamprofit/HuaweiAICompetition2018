from keras.models import load_model
from DataGenerator import DataGenerator
import numpy as np
import metrics
import sys
import cv2

IMG_SIZE = (1024, 1024)
INPUT_SIZE = (256, 256)

def printHelp():
    print('Usage: evaluate [path to model] [path to images]')

def reconstructImage(batches):
    nb_cols = np.ceil(IMG_SIZE[0] / INPUT_SIZE[0])
    nb_rows = np.ceil(IMG_SIZE[1] / INPUT_SIZE[1])

    assert nb_cols * nb_rows == len(batches)

    # Create an empty image of maximum size that
    # includes padding to fit all batches
    size_x = int(INPUT_SIZE[0] * nb_cols)
    size_y = int(INPUT_SIZE[1] * nb_rows)
    reconstruct = np.zeros((size_x, size_y, 3), np.uint8)

    for i in range(len(batches)):
        row = np.floor(i / nb_cols)
        col = (i - (row * nb_cols))

        x_min = int(col * INPUT_SIZE[0])
        y_min = int(row * INPUT_SIZE[1])
        x_max = int((col + 1) * INPUT_SIZE[0])
        y_max = int((row + 1) * INPUT_SIZE[1])

        # Paste the batch onto the image
        batch = batches[i]
        batch.reshape(INPUT_SIZE[0], INPUT_SIZE[1], 3)

        reconstruct[y_min:y_max, x_min:x_max] = batch[0][0:INPUT_SIZE[1],
                                                         0:INPUT_SIZE[0]]

    # Crop out the padding to get the original image
    original = reconstruct[0:IMG_SIZE[1], 0:IMG_SIZE[0]]
    return original

def reconstructImages(batches):
    nb_cols = np.ceil(IMG_SIZE[0] / INPUT_SIZE[0])
    nb_rows = np.ceil(IMG_SIZE[1] / INPUT_SIZE[1])
    batches_per_img = int(nb_cols * nb_rows)
    nb_images = int(len(batches) / batches_per_img)

    reconstructed = []
    for i in range(0, nb_images, batches_per_img):
        img = reconstructImage(batches[i : i+batches_per_img])
        reconstructed.append(img)

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

def evaluate(references, images):
    assert len(references) == len(images)

    for i in range(len(references)):
        res = metrics.psnr(references[i], images[i])
        print('PSNR: ' + str(res))

def main():
    modelPath  = sys.argv[1]
    imagesPath = sys.argv[2]

    datagen = DataGenerator(imagesPath, 1, IMG_SIZE, INPUT_SIZE)
    batches = datagen.getAllBatchesOfX()
    model   = load_model(modelPath)

    predictions   = getPredictions(model, batches)
    reconstructed = reconstructImages(predictions)
    print(str(len(reconstructed)) + ' images reconstructed')

    images = datagen.getAllX()
    evaluate(images, reconstructed)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        printHelp()
        sys.exit()

    main()
