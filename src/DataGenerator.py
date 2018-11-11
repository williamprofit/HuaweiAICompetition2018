import keras

from pathlib import Path
import cv2
import numpy as np
import random


class DataGenerator(keras.utils.Sequence):
    def __init__(self, path, batch_size):
        self.index = 0
        self.batch_size = batch_size

        self.paths, self.x_paths, self.y_paths = self.loadPaths(path)
        self.nbImages = len(self.x_paths)

    def __len__(self):
        return int(np.ceil(self.nbImages / self.batch_size))

    def __getitem__(self, index):
        i = self.index
        batch_size = self.batch_size

        # Get all image paths to load
        x_to_load = self.x_paths[i:i+batch_size]
        y_to_load = self.y_paths[i:i+batch_size]

        # Make sure we wrap around if we get to end of data list
        self.index = i + batch_size
        if self.index >= self.nbImages:
            nb_missing = batch_size - len(x_to_load)
            x_to_load = x_to_load + self.x_paths[0:nb_missing]
            y_to_load = y_to_load + self.y_paths[0:nb_missing]

            self.reset()

        # Load images
        x_train = []
        y_train = []
        for path in x_to_load:
            x_train.append(cv2.imread(path))
        for path in y_to_load:
            y_train.append(cv2.imread(path))

        # Convert lists to numpy arrays
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        # Preprocess images
        for j in range(len(x_train)):
            x_train[j] = self.preprocessImage(x_train[j])
            y_train[j] = self.preprocessImage(y_train[j])

        return (x_train, y_train)

    def reset(self):
        # Set index to 0 and reshuffle data
        self.index = 0
        self.shuffleLists(self.x_paths, self.y_paths)

    def preprocessImage(self, img):
        img = img.astype('float32')
        img = img / 255
        return img

    def shuffleLists(self, a, b):
        combined = list(zip(a, b))
        random.shuffle(combined)
        return zip(*combined)

    def loadPaths(self, path):
        paths   = list(Path(path).rglob('*.png'))
        x_paths = []
        y_paths = []
        for i in range(len(paths)):
            paths[i] = str(paths[i].resolve())

            if 'Clean' in paths[i]:
                x_paths.append(paths[i])
            elif 'Noisy' in paths[i]:
                y_paths.append(paths[i])
            else:
                print('!WARNING! Invalid image path: ' + pats[i]
                     + '. Image was not loaded.')

        x_paths.sort()
        y_paths.sort()

        # Shuffle the paths together
        x_paths, y_paths = self.shuffleLists(x_paths, y_paths)

        if len(x_paths) == len(y_paths):
            print(str(len(paths)) + ' paths loaded.')
        else:
            print('!CRIT WARNING! x_paths and y_paths are mismatched')

        return paths, x_paths, y_paths
