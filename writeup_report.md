[//]: # (Image References)
[image0]: ./results/sample_image.png "Example of Camera Image"
[image1]: ./results/gamma_shift.png "Example of Gamma Shifted Images"
[image2]: ./results/affine_transformation.png "Examples of Affine Image Transformations"
[image3]: ./results/orignal_steering_angle_distribution.png "Orignal Steering Angle Distribution"
[image4]: ./results/augmented_steering_angle_distribution.png "Augmented Steering Angle Distribution"

# **Behavioral Cloning Project**
---
Within this project a deep neural network is trained to clone the driving behavior of a human driver.
The car is driven within the Udacity car simulator. Three cameras are 'mounted' on the front of the car and collect left, right and center images during driving. The steering angle is the training objective and later gets predicted by the neural net. This makes this problem a regression problem.

The main **Goals of the Project** are:
* Use the simulator to collect data of "good" driving behavior
* Build, a convolution neural network with Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The main **New Techniques of the Project** for me are:
* Data augmentation
* Online batch generation

These points are addressed separately within this report.

#### Files of the Project
The project includes the following files:

* **model.py**: builds, trains and saves the model
* **pipeline.py**: preprocessing and batch generator functions
* **drive.py**: drives the trained model within the car simulator
* **video.py**: converts the images to a single video
* **video.mp4**: autonomous drive video on Track 1
* **model.h5**: model architecture and weights of the trained model
* **writeup_report.md**: description of the project

#### Comments on the Project
Unfortunately the car simulator did run on my local machine. After spending long hours trying to find and fix the problem and consulting the discussion forums, I decided to change the operating system on my PC from Ubuntu Mate (which does not support Unity) to Ubuntu. This solved the problem but consumed a lot of time that I planed to spend on the project itself.
Therefore, to my dissatisfaction, this project, although finished, lacks a certain depth at some places ...   

## Project Steps

### Data Analysis and Visualization
The images get recorded by the car simulator and are stored in a directory IMG.
In order to load the data a second file is saved called 'driving_log.csv' which contains the following data:

|center |left |right |steering |throttle |brake |speed
|---|---|---|---|---|---|---|
|/IMG/center_2017_04_09_20_52_29_816.jpg |/IMG/left_2017_04_09_20_52_29_816.jpg |/IMG/right_2017_04_09_20_52_29_816.jpg |0.000000 |0.317649 |0 |2.980154
|/IMG/center_2017_04_09_20_52_29_965.jpg |/IMG/left_2017_04_09_20_52_29_965.jpg |/IMG/right_2017_04_09_20_52_29_965.jpg |-0.186531 |0.000000 |0 |3.036884
|/IMG/center_2017_04_09_20_52_30_090.jpg |/IMG/left_2017_04_09_20_52_30_090.jpg |/IMG/right_2017_04_09_20_52_30_090.jpg |-0.557247 |0.000000 |0 |2.949280
|/IMG/center_2017_04_09_20_52_30_235.jpg |/IMG/left_2017_04_09_20_52_30_235.jpg |/IMG/right_2017_04_09_20_52_30_235.jpg |-0.474747 |0.000000 |0 |2.903985
|/IMG/center_2017_04_09_20_52_30_361.jpg |/IMG/left_2017_04_09_20_52_30_361.jpg |/IMG/right_2017_04_09_20_52_30_361.jpg |-0.100224 |0.000000 |0 |2.863618
|/IMG/center_2017_04_09_20_52_30_510.jpg |/IMG/left_2017_04_09_20_52_30_510.jpg |/IMG/right_2017_04_09_20_52_30_510.jpg |0.000000 |0.000000 |0 |2.816877
|/IMG/center_2017_04_09_20_52_30_626.jpg |/IMG/left_2017_04_09_20_52_30_626.jpg |/IMG/right_2017_04_09_20_52_30_626.jpg |0.000000 |0.349161 |0 |3.214293
|/IMG/center_2017_04_09_20_52_30_773.jpg |/IMG/left_2017_04_09_20_52_30_773.jpg |/IMG/right_2017_04_09_20_52_30_773.jpg |-0.440548 |0.000000 |0 |3.284635
|/IMG/center_2017_04_09_20_52_30_897.jpg |/IMG/left_2017_04_09_20_52_30_897.jpg |/IMG/right_2017_04_09_20_52_30_897.jpg |-0.069604 |0.000000 |0 |3.238564

It is important to distinguish between the different camera images because for the left and right images a steering angle correction is applied.
I followed the suggested value of +- 0.2 from the lecture which worked fine for me.

Here is an typical image from from the center camera of the simulator:

![alt text][image0]

In order to use these images for training the network several preprocessing and augmentation steps are required.

#### Brightness Augmentation
The first step is creating differently bright versions of the images via gamma shifting.
This helps the model to generalize under different lighting conditions.
![alt text][image1]
#### Flip Image
The next step is to flip the image and the steering angle randomly from left to right. This is a cheap way to increase the amount of training data.  
#### Steering Angle Augmentation
While flipping the images is helpful, it is not sufficient to even the distribution of samples w.r.t. the steering angle.
To better illustrate the problem here is a graph of the distribution of steering angles for the data I collected to train my model:

![alt text][image3]
As can be seen there basically only three different steering angles. This discrete distribution results from steering the car via the keyboard which allows only on-off steering. Now the problem is, that with this training data the model will almost always output one of three steering angles for any image it sees. However, if the car is at an slightly different position with a slightly different speed, choosing one of three angles will likely not be the right action the keep the car on the track.
Therefore the rational behind the next augmentation step is the even out the steering angle distribution. Ideally this means, that the car would have seen any for any possible image (at any brightness).
I adapted the idea from [here](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk) and i must admit, that I would not have come up with that idea myself. However, it works really well.

Here is the code for augmenting the steering angles:

```python
def random_affine_transformation(image, angle, shear_range=200):
    """
    The following code is adapted from:
    https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
    """
    rows, cols = image.shape[0:2]
    dx = np.random.randint(-shear_range, shear_range)
    random_point = [cols/2 + dx, rows/2]
    triangle1 = np.float32([[0,         rows],
                            [cols,      rows],
                            [cols/2,    rows/2]])
    triangle2 = np.float32([[0,    rows],
                            [cols, rows],
                            random_point])

    steering_correction = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    transf_matrix = cv2.getAffineTransform(triangle1, triangle2)
    #print(triangle2[2,0]-triangle1[2,0])
    #print(transf_matrix)
    image = cv2.warpAffine(image, transf_matrix, (cols, rows), borderMode=1)
    angle += steering_correction

    return image, angle
```
Appling this transformation to each image, the steering angle distribution becomes far more evenly distributed:
![alt text][image4]

To my experience, this augmentation step makes all the difference.
Here are some examples of randomly augmented images:
![alt text][image2]

### Data Proprocessing
The preprocessing steps are included as the first layers of the model. This is required to use the model with real time data to make predictions.
The preprocessing that is preformed within this project are normalization and cropping.
Both steps were already motivated in the lecture.
Here is how they are implemented with keras:
```python
init = Input(shape=(160, 320, 3))
x    = Cropping2D(cropping=((70, 25), (0, 0)))(init)
x    = Lambda(lambda x: x / 127.5 - 1.0)(x)
```

## Model Architecture
The model architecture is inspire by the paper from NVIDIA [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

```python
# The model is inspired by the NVIDIA paper "End to End Learning for Self-Driving Cars"
# Paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

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
```
It is a fully convolutional network. One thing to mention are the first two conv layers where I decided to use filters with the dimension (2,4). The rational behind this is, that I wanted the image dimension to become square, so that the vertical and horizontal dimension are represented equally.
The model worked quite well but I had no time to analyzed the effect of this design choice in detail.  

A dropout layer is added to the model in order to reduce overfitting.


Here is a summary of the model:

|Layer (type) | Output Shape | Param #|
|-------------------------------------|
|input_1 (InputLayer)|(None, 160, 320, 3)| 0|
|cropping2d_1 (Cropping2D)| (None, 65, 320, 3)|  0|
|lambda_1 (Lambda)|(None, 65, 320, 3)|  0|
|conv2d_1 (Conv2D)|(None, 65, 107, 16)| 400|
|conv2d_2 (Conv2D)|(None, 65, 54, 32)|  4128|
|conv2d_3 (Conv2D)|(None, 33, 27, 48)|  13872|  
|conv2d_4 (Conv2D)|(None, 17, 14, 64)|  27712|  
|conv2d_5 (Conv2D)|(None, 17, 14, 64)|  36928|  
|conv2d_6 (Conv2D)|(None, 9, 7, 64)| 36928|  
|flatten_1 (Flatten)| (None, 4032)|  0|
|dense_1 (Dense)|  (None, 100)|403300|
|dropout_1 (Dropout)| (None, 100)|0|
|dense_2 (Dense)|  (None, 10)| 1010|
|dense_3 (Dense)|  (None, 1)|  11|  

| Name | Value|
|-------------|------------|
|Total params |524,289.0|
|Trainable params| 524,289.0|
|Non-trainable params|0.0|



### Batch Generator
The car simulator engine saves the images with a dimension of (160, 320, 3) pixels. This is significantly larger than the datasets of previous projects like the MNIST dataset where the image dimension is only (28, 28, 1). Doing the math reveals that this time each image requires about 196 times as much disk space as for an MNIST image. Therefore, in order to train the model on a PC or an AWS EC2 instance, the batches have to be created online.
This means, that instead of storing the entire training and validation datasets on disk and feeding one batch at a time to the model, the images for one batches are only loaded and augmented when required.  

The following code shows the implementation used within this project:
```python
def generate_batch(batch_size, img_path, filename):
    """
    The following code is inspired by stackoverflow:
    http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python
    """
    cnt = 0
    while True:
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
```
Thanks to the great keras API, this approach can be implemented without great difficulty.

```python
import pipeline as pl

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
```
### Training the model
My first attempt was to train the model on an AWS EC2 instance and then use the trained model to drive the car autonomously on my local PC. While the training worked fine, the model could not run on my local machine. I think that this issue can be solved, but due to time constrains I decided to try training and testing the model locally.
For the local training I had to reduce the batch size, the steps per epoch and the number of epochs significantly.

However, this turned out (to my surprise!) to still work very well too.

The parameters I chose for the final training are:

```python
batch_size = 64
nb_epochs  = 20
steps_per_training_epoch = 5
steps_per_validation_epoch = np.max((1, int(0.1 * steps_per_training_epoch)) )

# Definition of the model architecture
...

optimizer = Adam()
model.compile(optimizer=optimizer, loss='mse')
```
For the Adam optimizer is kept the recommended default values for the hyper parameters.

### Results

The model is tested on the first (left) track in the car simulator.
The video of the test run can be seen
[![here](https://www.youtube.com/watch?v=lITnEx7hRW0&feature=youtu.be/0.jpg)](https://www.youtube.com/watch?v=lITnEx7hRW0&feature=youtu.be "Video Title").

It is also available in the repository as video.mp4.

### Discussion

This project was overall a great learning experience. To me the most interesting parts have been using a neural net to drive a car, data augmentation and online batch generation.

I was very surprised that the model could drive the car so well with little training.
To improve the model further it would be interesting to include the speed and brake data in the training and eventually come up with some augmentation for these.
Further I think that recurrent neural nets may provide some means to introduce past state information to the model.
Adding allowed driving ground information to the model would be very interesting too.
I'm excited to learn about further technique in the coming lectures!

Thanks to the Udacity Team for putting up such a great learning project.
