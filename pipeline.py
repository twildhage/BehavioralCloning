import numpy as np
import cv2

def gamma_correction(image):
    """
    A gamma correction is used to change the brightness of training images.
    The correction factor 'gamma' is sampled randomly in order to generated
    an even distrubtion of image brightnesses. This shall allow the model to
    generalize.
    The code is inspired by:
    http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    :image:
        Source image as numpy array
    :return:
        Gamma corrected version of the source image
    """
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
