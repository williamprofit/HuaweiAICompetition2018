import numpy
import math
import cv2

# Assumes 255 is the max pixel val
def psnr(reference, image):
    mse = numpy.mean( (image - reference) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
