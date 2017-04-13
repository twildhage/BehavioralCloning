import pipeline as pl


import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Input, Concatenate, Add
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Adam

nb_epochs  = 1
batch_size = 10
nb_validation_samples = 40
nb_training_samples_per_epoch = 100
nb_validation_samples = int(0.2 * nb_training_samples_per_epoch)


# The model is ispired by the NVIDIA paper "End to End Learning for Self-Driving Cars"
# Paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

init = Input(shape=(pl.IMAGE_SIZE, pl.IMAGE_SIZE, 3))
x    = Lambda(lambda x: x / 127.5 - 1.0)(init)

# Convolutional layers
x    = Conv2D(16, (3,3), activation='relu', padding='same', strides=(2,2) )(x)
x    = Conv2D(32, (3,3), activation='relu', padding='same', strides=(1,1) )(x)
x    = Conv2D(48, (3,3), activation='relu', padding='same', strides=(1,1) )(x)
x    = Conv2D(64, (3,3), activation='relu', padding='same', strides=(1,1) )(x)
x    = Conv2D(64, (3,3), activation='relu', padding='same', strides=(1,1) )(x)
x    = Conv2D(64, (3,3), activation='relu', padding='same', strides=(2,2) )(x)

# Fully connected layers
x    = Flatten()(x)
x    = Dense(100, activation='relu')(x)
x    = Dense(10, activation='relu')(x)

# Output without activation
out  = Dense(1)(x)

model = Model(init, out)
model.summary()
optimizer = Adam()
model.compile(optimizer=optimizer, loss='mse')


training_batch_generator = pl.generate_batch(batch_size,
                                              pl.DATA_PATH,
                                              pl.DRIVING_LOG_FILE)

validation_batch_generator = pl.generate_batch(nb_validation_samples,
                                                pl.DATA_PATH,
                                                pl.DRIVING_LOG_FILE)

history = model.fit_generator(training_batch_generator,
                              steps_per_epoch=batch_size,
                              epochs=nb_epochs,
                              validation_data=validation_batch_generator,
                              validation_steps=nb_validation_samples,
                              verbose=1)


# Save model architectur (*.json) and weights (*.h5)
pl.save_model(model)
