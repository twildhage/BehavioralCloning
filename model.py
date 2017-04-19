print("Import libraries ...")

import pipeline as pl


#import tensorflow as tf
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Lambda, Input, Concatenate, Add, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Adam
import pickle
print("Done.")

batch_size = 64 
nb_epochs  = 20 
steps_per_training_epoch = 5
steps_per_validation_epoch = np.max((1, int(0.1 * steps_per_training_epoch)) )



# The model is ispired by the NVIDIA paper "End to End Learning for Self-Driving Cars"
# Paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

#init = Input(shape=(pl.IMAGE_SIZE, pl.IMAGE_SIZE, 3))
init = Input(shape=(160, 320, 3))
x    = Cropping2D(cropping=((70, 25), (0, 0)))(init)
x    = Lambda(lambda x: x / 127.5 - 1.0)(x)

# Convolutional layers
x    = Conv2D(16, (2,4), activation='relu', padding='same', strides=(1,3) )(x)
x    = Conv2D(32, (2,4), activation='relu', padding='same', strides=(1,2) )(x)
x    = Conv2D(48, (3,3), activation='relu', padding='same', strides=(2,2) )(x)
x    = Conv2D(64, (3,3), activation='relu', padding='same', strides=(2,2) )(x)
x    = Conv2D(64, (3,3), activation='relu', padding='same', strides=(1,1) )(x)
x    = Conv2D(64, (3,3), activation='relu', padding='same', strides=(2,2) )(x)

# Fully connected layers
x    = Flatten()(x)
x    = Dense(100, activation='relu')(x)
x    = Dropout(0.5)(x)
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

validation_batch_generator = pl.generate_batch(batch_size,
                                                pl.DATA_PATH,
                                                pl.DRIVING_LOG_FILE)

print("Start training the model ...")
history = model.fit_generator(training_batch_generator,
                              steps_per_epoch=steps_per_training_epoch,
                              epochs=nb_epochs,
                              validation_data=validation_batch_generator,
                              validation_steps=steps_per_validation_epoch,
                              verbose=1)


model.save("model.h5")
#with open('history.p', 'wb') as f:
#    pickle.dump(history, f)
