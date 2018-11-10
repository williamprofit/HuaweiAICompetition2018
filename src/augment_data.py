import sys
import os
from pathlib import Path
import cv2


# Loads images separated in list of Clean and Noisy.
# The images are themselves tuples containing the image
# itself and its name for later saving purposes
def loadImages(paths):
    images_x = []
    images_y = []

    for p in paths:
        p = str(p.resolve())

        print('Loading ' + p)

        # Images are tuples with data and name
        img  = cv2.imread(p)
        name = p.split('/')[-1][:-4]
        if 'Clean' in p:
            images_x.append((img, name))
        elif 'Noisy' in p:
            images_y.append((img, name))
        else:
            print('!WARNING! Invalid image path: ' + p
                 + '. Image was not loaded.')

    return (images_x, images_y)

# Given a list of images, augments them in multiple ways and
# adds a postfix specific to each transformation to the name
def augmentImages(images):
    print('Augmenting ...')

    new_x = []
    new_y = []

    for x in images[0]:
        new_x.append(flipHorizontally(x))
        new_x.append(flipVertically(x))
        new_x.append(flipBothWays(x))
    for y in images[1]:
        new_y.append(flipHorizontally(y))
        new_y.append(flipVertically(y))
        new_y.append(flipBothWays(y))

    return (new_x, new_y)

def flipHorizontally(img, postfix='_rot_0'):
    newImg  = cv2.flip(img[0], 0)
    newName = img[1] + postfix
    return (newImg, newName)

def flipVertically(img, postfix='_rot_1'):
    newImg  = cv2.flip(img[0], 1)
    newName = img[1] + postfix
    return (newImg, newName)

def flipBothWays(img, postfix='_rot_2'):
    newImg  = cv2.flip(img[0], 0)
    newImg  = cv2.flip(newImg, 1)
    newName = img[1] + postfix
    return (newImg, newName)

# Saves all the given images in the specified path. Clean images
# will be saved in subdirectory Clean and same for Noisy. Images
# will be saved with their respective postfix attached.
def saveImages(images, path):
    print('Saving augmented images ...')

    createSavingDirs(path)

    for x in images[0]:
        cv2.imwrite(path + '/Clean/aug_' + x[1] + '.png', x[0])
    for y in images[1]:
        cv2.imwrite(path + '/Noisy/aug_' + y[1] + '.png', y[0])

def createSavingDirs(path):
    dirs = ['/Clean', '/Noisy']
    for d in dirs:
        if not os.path.isdir(path + d):
            os.makedirs(path + d, exist_ok=True)

def printBatchNb(start, size, batchSize):
    print('-- Batch ' + str(int(start/batchSize)+1) + '/'
         + str(int(size / batchSize)) + ' --')

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = '../sample_data/'

    paths = list(Path(path).rglob('*.png'))

    batchSize = 10
    for start in range(0, len(paths), batchSize):
        printBatchNb(start, len(paths), batchSize)

        end = start + batchSize
        if end > len(paths):
            end = len(paths)

        batch = paths[start:end]

        images = loadImages(batch)
        images = augmentImages(images)
        saveImages(images, '../sample_data/augmented/')

if __name__ == '__main__':
    main()
