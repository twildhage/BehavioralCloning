import numpy as np
import cv2
import pandas as pd
from scipy.stats import bernoulli
#from sklearn.utils import shuffle

def gamma_correction(image, mode='random', gamma=1.25):
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
    if mode == 'random':
        gamma_ = np.random.uniform(0.4, 1.5)
    elif mode == 'manual':
        gamma_ = gamma
    else:
        print('mode has to be random or manual')
        return 0
    inv_gamma = 1.0 / gamma_
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def get_consistent_modification_of(image, angle):
    """
    some description
    :image
        sss
    :angle
        sss
    :return
        XXX
    """
    flip_the_image = bernoulli.rvs(0.5)
    if flip_the_image == True:
        return np.fliplr(image), -angle
    else:
        return image, angle

def generate_random_subset_of_dataset(subset_size, nb_subsets, datafile):
    camera_names     = ('center', 'left', 'right')
    cameras          = dict({key:val for key,val in enumerate(camera_names)})
    steering_offsets = dict({key:val for key,val in zip(camera_names, [0, 0.2, -0.2])})
    dataset          = pd.read_csv(datafile)
    stop_generator   = False
    while stop_generator==False:
        for i in range(nb_subsets):
            # Shuffle dataset, so that the first n entries represent a random subset
            print(dataset)
            np.random.shuffle(dataset)
            print(dataset)
            # Create a list with random cameras: ['left', 'left', 'center', 'right', ...]
            random_cameras = [cameras[i] for i in np.random.randint(0, 3, subset_size)]
            # Create the images files and angles
            img_files = [dataset[camera][i] for i, camera in enumerate(random_cameras)]
            angles    = [dataset['steering'][i] + steering_offsets[camera] for i, camera in enumerate(random_cameras)]
            yield img_files, angles
        stop_generator = True




def generate_batch(batch_size, path, file_name):
    """
    This function generates a generator, which then yields a training batch.
    If this sounds confusing, check out this excellent explanation on
    stackoverflow:
    http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python
    """
#    X_batch = []
#    y_batch = []
    for i in range(batch_size):
        pass



