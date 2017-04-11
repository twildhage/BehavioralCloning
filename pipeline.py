import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from sklearn.utils import shuffle


DATA_PATH = '/home/timo/Documents/mldata/car_sim_video_images/training_data/'
DRIVING_LOG_FILE = 'driving_log.csv'


def random_gamma_shift(image, mode='random', gamma=1.25):
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


def random_flip(image, angle):
    flip_the_image = bernoulli.rvs(0.5)
    if flip_the_image == True:
        return np.fliplr(image), -angle
    else:
        return image, angle

def cut(image, upper, lower):
    height = image.shape[0]
    l = int(height * upper)
    u = int(height * (1-lower))
    print(l, u)
    return image[l:u,::]

def resize(image, height, width):
    return cv2.resize(image, (height, width))


def get_consistent_modification_of(image, angle):
    """
    some description
    :image
        image from one camera position (left, right or center)
    :angle
        steering angle
    :return
        modified image and angle
    """
    image, angle = random_flip(image, angle)
    image        = random_gamma_shift(image)
    image        = cut(image, 0.38, 0.15)
    image        = resize(image, 48, 48)
    return image, angle


def get_random_subset_of_dataset(subset_size, filename):
    camera_names     = ('center', 'left', 'right')
    # Create dictionary: {0:'center', ...}
    cameras          = dict({key:val for key,val in enumerate(camera_names)})
    # Create dictionary: {'center':0.0, ...}
    steering_offsets = dict({key:val for key,val in zip(camera_names, [0.0, 0.2, -0.2])})

    dataset          = pd.read_csv(filename)
    # Shuffle dataset, so that the first n entries represent a random subset
    dataset = shuffle(dataset)
    # Create a list with random cameras: ['left', 'left', 'center', 'right', ...]
    random_cameras = [cameras[i] for i in np.random.randint(0, 3, subset_size)]
    # Create the images files and angles
    img_files = [dataset[camera].iloc[i] for i, camera in enumerate(random_cameras)]
    angles    = [dataset['steering'].iloc[i] + steering_offsets[camera] for i, camera in enumerate(random_cameras)]
    return img_files, angles




def generate_batch(batch_size, nb_batches, img_path, filename):
    """
    This function generates a generator, which then yields a training batch.
    If this sounds confusing, check out this excellent explanation on
    stackoverflow:
    http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python
    """
    cnt = 0
    while cnt < nb_batches:
        X_batch = []
        y_batch = []
        img_files, angles = get_random_subset_of_dataset(batch_size, (img_path + filename))
        for img_file, angle in zip(img_files, angles):
            img = plt.imread(img_path + img_file)
            # Modify images
            img, angle = get_consistent_modification_of(img, angle)
            X_batch.append(img)
            y_batch.append(angle)
        yield np.array(X_batch), np.array(y_batch)
        cnt += 1