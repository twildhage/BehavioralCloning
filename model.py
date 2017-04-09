import pipeline

import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Input, Concatenate, Add
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Adam

nb_epochs = 10
nb_training_samples_per_epoch = 100
nb_validation_samples = int(0.2 * nb_training_samples_per_epoch)




# The model is ispired by the NVIDIA paper "End to End Learning for Self-Driving Cars"
# Paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

init = Input(shape=(32, 32, 3))
x    = Lambda(lambda x: x / 127.5 - 1.0)(init)

# Convolutional layers
x    = Conv2D(16, (3,3), activation='relu', padding='same', strides=(2,2) )(x)
x    = Conv2D(32, (3,3), activation='relu', padding='same', strides=(1,1) )(x)
x    = Conv2D(48, (3,3), activation='relu', padding='same', strides=(1,1) )(x)
x    = Conv2D(64, (3,3), activation='relu', padding='same', strides=(1,1) )(x)
x    = Conv2D(64, (3,3), activation='relu', padding='same', strides=(1,1) )(x)
x    = Conv2D(64, (3,3), activation='relu', padding='same', strides=(2,2) )(x)

# Fully connected layers
x    = Dense(100, activation='relu')(x)
x    = Dense(10, activation='relu')(x)

# Output without activation
out  = Dense(1)(x)

model = Model(init, out)
model.summary()

