import keras

from pathlib import Path
import cv2
import numpy as np
import random

class DataGenerator(keras.utils.Sequence):

    def __init__(self, path, batch_size, image_size, input_size, permutations=False):
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
        self.permutations = permutations

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

        nb_images_needed = int(np.ceil((end - start) / self.inputs_per_image)) + 1
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

    def getImageAugments(self, image):
        images = []
        for i in range(4):
            M = cv2.getRotationMatrix2D((self.input_size[0]/2,self.input_size[0]/2),-i*90,1)
            dst = cv2.warpAffine(image,M,self.input_size)
            images.append(dst)

        image = cv2.flip(image, 1)
        for i in range(4):
            M = cv2.getRotationMatrix2D((self.input_size[0]/2,self.input_size[0]/2),-i*90,1)
            dst = cv2.warpAffine(image,M,self.input_size)
            images.append(dst)

        return images

    def getImageDeaugment(self, images):
        deaugImages = []
        for i in range(int(len(images)/2)):
            M = cv2.getRotationMatrix2D((int(self.input_size[0]/2),int(self.input_size[0]/2)),i*90,1)
            dst = cv2.warpAffine(images[i],M,self.input_size)
            deaugImages.append(dst)

        for i in range(int(len(images)/2),len(images)):
            M = cv2.getRotationMatrix2D((int(self.input_size[0]/2),int(self.input_size[0]/2)),i*90,1)
            dst = cv2.warpAffine(images[i],M,self.input_size)
            dst = cv2.flip(dst,1)
            deaugImages.append(dst)

        retImage = np.zeros((self.input_size[1], self.input_size[0], 3), np.float)
        for image in deaugImages:
            avgImg = np.array(image,dtype=np.float)
            retImage = retImage+avgImg/len(deaugImages)

        retImage = np.array(np.round(retImage),dtype=np.uint8)
        return retImage
    
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

            if(col != self.nb_cols-1):
                x_min = int(col * self.input_size[0])
            else:
                x_min = self.image_size[0] - self.input_size[0]

            if(row != self.nb_rows-1):
                y_min = int(row * self.input_size[1])
            else:
                y_min = self.image_size[1] - self.input_size[1]

            x_max = x_min + self.input_size[0]
            y_max = y_min + self.input_size[1]

            # Fetch the image containing the input
            img_nb = int(np.floor(i / self.inputs_per_image))
            img = images[img_nb]

            # Fill the subimage with the image
            subimage = img[y_min : y_max, x_min : x_max]

            if(self.permutations):
                augImages = self.getImageAugments(subimage)
                batch.extend(augImages)
            else:
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

    def getAllX(self):
        images = []
        print("path: " , self.x_paths)
        for path in self.x_paths:
            images.append(cv2.imread(path))

        return images

    def getAllY(self):
        images = []
        for path in self.y_paths:
            images.append(cv2.imread(path))

        return images

    def getAllBatchesOfX(self):
        original_batch_size = self.batch_size
        self.batch_size     = self.nb_batches

        images, _ = self.getImagesForBatch(0)
        xs = self.createBatches(0, self.nb_batches, images)

        for i in range(len(xs)):
            xs[i] = self.preprocessImage(xs[i])

        xs = np.asarray(xs)

        self.batch_size = original_batch_size

        return xs

    def getAllBatchesOfY(self):
        original_batch_size = self.batch_size
        self.batch_size     = self.nb_batches

        _, images = self.getImagesForBatch(0)
        ys = self.createBatches(0, self.nb_batches, images)

        for i in range(len(ys)):
            ys[i] = self.preprocessImage(ys[i])

        ys = np.asarray(ys)

        self.batch_size = original_batch_size

        return ys

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

            if 'Noisy' in paths[i]:
                x_paths.append(paths[i])
            elif 'Clean' in paths[i]:
                y_paths.append(paths[i])
            else:
                x_paths.append(paths[i])

        x_paths.sort()
        y_paths.sort()

        if(len(y_paths)==0):
            y_paths = x_paths

        if len(x_paths) == len(y_paths):
            # Shuffle the paths together
            x_paths, y_paths = self.shuffleLists(x_paths, y_paths)
            print(str(len(paths)) + ' paths loaded.')
        else:
            print('!CRIT WARNING! x_paths and y_paths are mismatched')

        return paths, x_paths, y_paths
