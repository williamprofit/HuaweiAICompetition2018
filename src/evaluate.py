from keras.models import load_model
from DataGenerator import DataGenerator
from skimage.measure import compare_psnr, compare_ssim
from denoise import denoise
import numpy as np
import metrics
import sys
import cv2

INPUT_SIZE = (256, 256)

def printHelp():
    print('Usage: evaluate [path to model] [path to images] {path to save dir}')

def evaluate(references, images):
    assert len(references) == len(images)

    print('Evaluating images ...')

    psnrTot = 0
    ssimTot = 0
    for i in range(len(references)):
        psnr = compare_psnr(references[i], images[i])
        ssim = compare_ssim(references[i], images[i], multichannel = True)
        ssimTot += ssim
        psnrTot += psnr
        print('PSNR: ' + str(psnr))
        print('SSIM: ' + str(ssim))

    psnrAvg = psnrTot/len(references)
    ssimAvg = ssimTot/len(references)
    print("Avg PSNR: " , psnrAvg)
    print("Avg SSIM: " , ssimAvg)

def main(modelPath, imagesPath, savePath):
    datagen = DataGenerator(imagesPath, 1, INPUT_SIZE)

    predictions = denoise(modelPath, imagesPath, savePath)
    references  = datagen.getAllY()

    evaluate(references, predictions)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        printHelp()
        sys.exit()
    elif len(sys.argv) == 4:
        savePath = sys.argv[3]
    else:
        savePath = ''

    main(sys.argv[1], sys.argv[2], savePath)
