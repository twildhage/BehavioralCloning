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

* **model.py**: build, train and save the model
* **pipeline.py**: preprocess the datasets and definition of batch generator
* **drive.py**: drive the trained model within the car simulator
* **video.py**: convert the images to a single video
* **model.h5**: model architecture and weights of the trained model
* **writeup_report.md**: description of the project

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

Here is an typical image from the simulator:

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
As can be seen there basically only three different steering angles. This discrete distribution results from steering the car via the keyboard which allows only on-off steering. No the problem is, that with this training data the model will almost always output one of three steering angles for any image it sees. However, if the car is at an slightly different position with a slightly different speed, choosing one of three angles will likely not be the right action the keep the car on the track.
Therefore the rational behind the next augmentation step is the even out the steering angle distribution. Ideally this means, that the car would have seen any for any possible image (at any brightness).
I adapted the idea from [here](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk) and i must admit, that I would not have come up with that idea myself. However, it works really well!

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
It is a fully convolutional network. One thing to mention are the first two conv layers where I decided to use filters wit the dimension (2,4). The rational behind this is, that I wanted the image dimension to become square, so that the vertical and horizontal dimension are represented equally.
The model worked quite well but I had no time to analyzed the effect of this design choice in detail.  

A dropout layer is added to the model in order to reduce overfitting.


Here is a summary of the model

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
In the lowest resolution the images have a dimension of (160, 320, 3) pixel. This is significantly larger than the datasets of previous projects like the MNIST dataset where the image dimension is only (28, 28, 1). Doing the math reveals that this time each image requires about 196 times as much disk space. In order to train the model on



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
