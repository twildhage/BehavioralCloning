# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains the source files for the Behavioral Cloning Project.

In this project, I use deep neural networks and convolutional neural networks to clone driving behavior. The training, validation and testing is done using Keras. The model outputs a steering angle to the virtual autonomous vehicle.

### Dependencies
This project requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) which includes Anaconda and Tensorflow.

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The simulator is created by Udacity for the Self-Driving-Car course. There is git repository with the source code and instructions on how to setup the [Udacity Simulator](https://github.com/udacity/self-driving-car-sim) for Windows, Linux and Mac.

### Files of the Project
The project includes the following files:

* **model.py**: builds, trains and saves the model
* **pipeline.py**: preprocessing and batch generator functions
* **drive.py**: drives the trained model within the car simulator
* **video.py**: converts the images to a single video
* **video.mp4**: autonomous drive video on Track 1
* **model.h5**: model architecture and weights of the trained model
* **writeup_report.md**: description of the project

### Training the Model

The model can be trained from the command line:

```python
python model.py
```
The results will be saved to *model.h5*.
The training was performed on my local machine (*Intel(R) Core(TM) i5 CPU M520 @2.40GHz 8GByte RAM*) for 20 epochs with 5 steps per epoch and a batch size of 64. Further details can be found in *writeup_report.md*.


### Run the Model on a Test Track
In order to run the pretrained model, download the car simulator and start it in autonomous mode. Clone or download the repository. Then execute the following code in your terminal:

```python
python drive.py model.h5
```

### Style Guides
Within this project I follow the [Udacity Git Commit Message Style Guide](https://udacity.github.io/git-styleguide/).
