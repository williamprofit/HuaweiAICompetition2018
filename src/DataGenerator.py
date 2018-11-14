import keras

from pathlib import Path
import cv2
import numpy as np
import random


class DataGenerator(keras.utils.Sequence):
    def __init__(self, path, batch_size, image_size, input_size):
        self.index = 0 # Indicates the current input
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_size = input_size

        self.nb_cols = np.ceil(image_size[0] / input_size[0])
        self.nb_rows = np.ceil(image_size[1] / input_size[1])

        self.inputs_per_image = self.nb_rows * self.nb_cols

        self.paths, self.x_paths, self.y_paths = self.loadPaths(path)

        self.nb_images  = len(self.x_paths)
        self.nb_inputs  = self.nb_images * self.inputs_per_image
        self.nb_batches = int(np.ceil(self.nb_inputs / self.batch_size))

    # Return number of batches per epoch
    def __len__(self):
        return self.nb_batches

    def getImagesForBatch(self, index):
        start = index
        end   = index + self.batch_size

        # Reset if end of epoch
        if end >= self.nb_inputs:
            end = self.nb_inputs - 1
            self.reset()

        nb_images_needed = int(np.ceil((end - start) / self.inputs_per_image))
        cur_img          = int(np.floor(start / self.inputs_per_image))

        x_to_load = self.x_paths[cur_img : cur_img + nb_images_needed]
        y_to_load = self.y_paths[cur_img : cur_img + nb_images_needed]

        # Load images
        x_images = []
        y_images = []
        for path in x_to_load:
            x_images.append(cv2.imread(path))
        for path in y_to_load:
            y_images.append(cv2.imread(path))

        return x_images, y_images

    def createBatches(self, start, end, images):
        fst_image_index = np.floor(start / self.inputs_per_image)
        top_left_index  = fst_image_index * self.inputs_per_image

        # Convert start and end from absolute to being relative to the top left
        # input of the first given image
        start = int(start - top_left_index)
        end   = int(end - top_left_index)

        batch = []
        for i in range(start, end, 1):
            # Determine the row and column of the input with index i
            row = np.floor(i / self.nb_cols) % self.nb_rows
            col = (i - (row * self.nb_cols)) % self.nb_cols

            # Convert row and column to pixel region. Note that for max, we
            # use min to make sure we don't go beyond the image size
            x_min = int(col * self.input_size[0])
            y_min = int(row * self.input_size[1])
            x_max = min(self.image_size[0] - x_min, self.input_size[0])
            y_max = min(self.image_size[1] - y_min, self.input_size[1])

            # Fetch the image containing the input
            img_nb = int(np.floor(i / self.inputs_per_image))
            img = images[img_nb]

            # Start with a blank subimage
            subimage = np.zeros((self.input_size[0], self.input_size[1], 3),
                                 np.uint8)

            # Fill the subimage with the image
            subimage[0 : y_max, 0 : x_max] = img[y_min : y_min + y_max,
                                                 x_min : x_min + x_max]

            batch.append(subimage)

        return batch

    def __getitem__(self, index):
        start = self.index
        end   = self.index + self.batch_size

        # Load images
        x_images, y_images = self.getImagesForBatch(start)

        # Create batches
        x_train = self.createBatches(start, end, x_images)
        y_train = self.createBatches(start, end, y_images)

        self.index = self.index + self.batch_size

        # Preprocess images
        for j in range(len(x_train)):
            x_train[j] = self.preprocessImage(x_train[j])
            y_train[j] = self.preprocessImage(y_train[j])

        # Convert lists to numpy arrays
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

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
                print('!WARNING! Invalid image path: ' + paths[i]
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
