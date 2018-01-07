# Behavioral Cloning

## Sara Collins

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_pictures/placeholder.png "Model Visualization"
[image2]: ./writeup_pictures/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* test directory: Includes testing configuration, testing examples and other testing artifacts. 
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* local_configuration.json for providing hyperparameters, file paths for input and output, and other configurable settings for training the model. 
* model.hdf5 containing a trained convolution neural network 
* test_model.py for unit tests for the model.py file. NOTE: This file is incomplete for this particular project, 
but as I've learned more and practiced with TDD a bit more, subsequent projects may have more comprehensive testing. 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.hdf5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments 
to explain how the code works where pertinent.  
  
I also practiced some of what I've learned about test-driven development in the past year where 
applicable to develop the training functions, and also attempted to create intuitively named 
functions. These should also help with code readability. 

### Model Architecture  

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with five convolutional
layers, and four dense layers(model.py lines 117-160) 

The layers are set up like this:

| Layer                  | Code Line(s)|  Size   | Filter   | Comment                                 |
|:----------------------:|:-----------:|:-------:|:--------:|:---------------------------------------:|
| Lambda - Normalization | 137         |         |          | Lambda layer to normalize image values. |
| Cropping - 2D          | 138         |         |          | Cropped to 25x70                        |
| 1 - Convolutional Layer| 139         | 24      | 5x5      | ReLU Activation                         |
| 2 - Convolutional Layer| 140         | 36      | 5x5      | ReLU Activation                         |
| 3 - Convolutional Layer| 141         | 48      | 5x5      | ReLU Activation                         |
| 4 - Convolutional Layer| 142         | 64      | 3x3      | ReLU Activation                         |
| 5 - Convolutional Layer| 143         | 64      | 3x3      | ReLU Activation                         |
| Dropout                | 144         |         |          | 25% Dropout                             |
| Flatten                | 145         |         |          | Flatten layers in prepration for dense  |
| 1 - Dense Layer        | 146         | 100     |          | ReLU Activation                         |
| 2 - Dense Layer        | 147         | 50      |          | ReLU Activation                         |
| 3 - Dense Layer        | 148         | 10      |          | ReLU Activation                         |
| 4 - Dense Layer        | 149         | 1       |          | No activation (needs to be linear for regression)|

Optimizer: Adam, with starting learning rate of 0.001. 
 
 
#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer between the convolutional layers and the dense layers in order to reduce overfitting. (model.py line 144). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 522-525). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually except for starting learning rate (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Training Strategy  

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
