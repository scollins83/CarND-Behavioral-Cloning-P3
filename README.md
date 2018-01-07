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

[model_architecture]: ./writeup_pictures/model_arch_diagram.jpg "Model Architecture"
[validation_loss_comparisons]: ./writeup_pictures/ValidationLossLastRuns.jpeg "Validation Loss Comparison of Last Runs"
[center_lane_simple]: ./writeup_pictures/center_line_driving_example.jpg "Center Line Simple"
[center_lane_complex]: ./writeup_pictures/complex_track_center_line.jpg "Center Line Complex"
[red_line_recover_1]: ./writeup_pictures/red_line_recovery_step_1.jpg "Red Line Recovery Image - Step 1"
[red_line_recover_2]: ./writeup_pictures/red_line_recovery_step_2.jpg "Red Line Recovery Image - Step 2"
[red_line_recover_3]: ./writeup_pictures/red_line_recovery_step_3.jpg "Red Line Recovery Image - Step 3"
[dirt_track_1]: ./writeup_pictures/dirt_track_step_1.jpg "Dirt Track - Step 1"
[dirt_track_2]: ./writeup_pictures/dirt_track_step_2.jpg "Dirt Track - Step 2"
[dirt_track_3]: ./writeup_pictures/dirt_track_step_3.jpg "Dirt Track - Step 3"
[flipped]: ./writeup_pictures/red_line_1_flipped.jpg "Red Line - Flipped"
[brightness]: ./writeup_pictures/center_brightness.jpg "Brightness"
[translation]: ./writeup_pictures/center_translation.jpg "Translation"
[shadow]: ./writeup_pictures/center_shadow.jpg "Shadow"
[mp4_thumbnail]: ./model_video/2018_01_07_04_17_16_598.jpg "Mp4 video thumbnail"

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

The model architecture used was adapted from the one proposed in NVIDIA Corporation's Bojarski, M. et al's "End to End Learning for Self-Driving Cars" [paper](https://arxiv.org/pdf/1604.07316v1.pdf "End to End Learning for Self-Driving Cars Paper") and explained in their [blog post](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/ "End to End Learning for Self-Driving Cars Blog Post"). However, the images were NOT converted to YUV image space (I experimented with that, but it didn't appear to give much lift and the car drove very slightly worse, with no discernable change in validation loss), and the dense layer of 1164 neurons was eliminated because it didn't appear to be necessary (no discernable change in car driving performance). 

The layers are set up like this:

![alt text][model_architecture]

Optimizer: Adam, with starting learning rate of 0.001. The model performance was largely dependent on learning rate, so I did tune the starting learning rate, which will be discussed later in this writeup. 
 
 
#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer between the convolutional layers and the dense layers in order to reduce overfitting. (model.py line 144). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 522-525). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually except for starting learning rate (model.py line 25). Learning rate made a huge difference in how the model performed, and while a final starting learning rate of 0.001 was selected, as I experimented with different augmentation methods, in some cases a starting learning rate of 0.0001 or 0.0005 performed better. 
Batch size was also critical to model training performance. On my local configuration with no GPU, I used a batch size of 64, but when I trained on floydhub with a GPU, I used a batch size of 256... 512 overflowed the 12GB of memory on the Tesla K80 for their regular GPU instances. 128 worked better than 256 in some cases as different experiments commences, as evidenced by checking training loss vs. validation loss in TensorBoard. 

Note: There were over 60 runs total, and I have tensorboard logs for all of those, but displayed are the last 3:
![alt text][validation_loss_comparisons]

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
The local set I used that resulted in the final model had three laps of center lane driving running clockwise on the simple track, three laps running counterclockwise on the simple track, and after seeing how the car handled different situations on the track, I added one recovery sample from the red/white striped line and eventually three samples of sufficiently performing the turn around the first curve after the bridge, so the car would stay on the track rather than veering off onto the dirt path. 
Interestingly enough, although I did not include samples of driving on the dirt track, in a few of the later model iterations where the car would veer off onto the dirt track, the car had learned sufficiently to be able to navigate the dirt track, and got back on the pavement at the end of it to successfully complete the lap. 

When training on floydhub, my [dataset](https://www.floydhub.com/scollins/datasets/simulator_data "Driving simulation data") included the same samples as the 'lightweight' version from my local setup mentioned in the preceding paragraph without the additional three dirt track corner samples, as they were added to my local sample after uploading the dataset to floyd. Additionally, the larger floyd dataset included: two laps of the complex track clockwise, two laps of the complex track counterclockwise, and recovery samples of the following: outer and inner red/white line, outer and inner bridge, outer and inner normal line, outer and inner curb going onto bridge, and from grass right next to the road from the complex track.
An earlier version of the floyd dataset also included backups and recovery from all sorts of other situations, including being completely off in the grass, navigating the dirt lane, running into a pole on the complex track, and ending up in the lake, but those proved to be detrimental to training this model so they were removed from the training set. 

### Training Strategy  

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the very simple model demonstrated in the presentation of the project, and then increase the complexity. I stopped after the NVIDIA architecture mentioned during the project walkthrough video, as it seemed to be sufficient for my needs, although I saw other sufficient architectures noted, particularly in a data augmentation blog post recommended to me by my Udacity mentor Rahul that was written by former Udacity Self-Driving Car Nanodegree alum Vivek Yadav, in his [blog post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9 "Using augmentation to mimic human driving") "Using Augmentation To Mimic Human Driving". 

In using the initial model presented in the regular Udacity course content, my car started to drive, but drove around in circles or would only go a little way and get off the road and get stuck. 

Then, I added more complex structures, approaching the NVIDIA-proposed architecture. This resulted in the car staying on the road quite nicely, albeit noticably avoiding driving on shadows. However, although the center line driving was usually perfect, the car would invariably (and frustratingly) miss the first curve after the bridge and would drive on the dirt track. As I continued to try to tune and add a few examples of turning that corner successfully to my training set, the model learned to navigate the dirt track 'successfully' and was able to get back on the pavement to finish the lap, but was still going off on the dirt track in the first place. 

I asked my Udacity mentor, Rahul, for some assistance, and began experimenting with adding different types of image augmentation. The model did ok and ALMOST rounded the corner after just flipping the image that was presented in the project walkthrough video, but wasn't quite there yet. 

I noted better loss values as I added more types of augmentation, but the car still went to the dirt track every time. 

Rahul had also noted that despite offering lots of clockwise and counter clockwise data, my data was unbalanced with dominance zero and low-steering-angle heavy, and recommended undersampling some of those measurements. Thus, I used the [imbalanced-learn package](https://github.com/scikit-learn-contrib/imbalanced-learn) to proportionately undersample lines with measurements ranging from -.1 to +.1, and only keeping 90% of the data from that range. The functions that accomplish this are called in model.py line 513, and the functions that accomplish this are in model.py lines 409 - 491 and have appropriate unit tests. 

Last but not least, the combination of downsampling and getting my training generator to best handle the data ended up resulting in the final model that completed the lap successfully, and even my 'lightweight' local version of my model trained on my Macbook Pro's CPU was sufficient. Initially, I had set up my generator to apply batch size only to the lines, and by the time I decided to complete all of the augmentation, it was multiplying that by eight for the number images going into each batch. So, I instead modified my 'generate full augment from lines' function (starting model.py line 357) to divide the batch size by eight, which was the number of images used for each line after augmentation (model.py line 373), and that finally resulted in the model that would stay on the track. 
Another augmentation technique used was offsetting the two side images, left and right, to make them appear to the model as being recovery image. The value I landed on for that was +/- 0.25 depending whether the image was left or right. (idea implemented from Yadav blog post cited earlier). 

Throughout training, I used a TensorBoard callback to monitor training loss and validation loss to check for overfitting. It was through this that I noted while the model was often presented with a small number of epochs (usually 5 - 10 from the Udacity content videos), the convergence of the model appeared to perhaps go past this and didn't appear to split to overfitting until after about 25 epochs in some of the configurations I tried.

Not wanting to potentially miss out on a better model, but not wanting to spend an unnecessary amount of time training either, I ended up setting up additional callbacks:
* Checkpointing (model.py line 535): I checkpointed the model after each epoch, but only saved the checkpoint if the model improved, guaranteeing I would still have the best model from each improvement point if the epochs really did indeed go too far.
* Early Stopping (model.py line 539): I included early stopping to stop training the model after 7 epochs of no improvement. 
* Reduce Learning Rate On Plateau (model.py line 543): While this may have given potential to interfere with the Adam optimizer, I set up the learning rate to reduce by a factor of .1 after the model had not improved for 4 epochs. This seemed to be useful in some configurations I tried, but was not useful in the final model iteration. 

#### 2. Final Model Architecture

See model architecture section. 

The final model occurred from a checkpoint taken at Epoch 3. 

Here is the video generated of the car driving a successful lap using the 'video.py' script included with the project.
Unfortunately, the video files were too large to commit to git gracefully, but I uploaded the videos to youtube.

**MP4 Video from Video.py, driving perspective:**
[![alt text](https://i.ytimg.com/vi/tFMG-lT_z2M/3.jpg)](https://www.youtube-nocookie.com/embed/tFMG-lT_z2M?rel=0)


Hyperparameters used, and can be found in file 'local_configuration.json'

| Hyperparameter | Value   |
|:--------------:|:-------:|
| Loss Function  | MSE     |
| Epochs         | 50 (checkpointed - used at epoch 3, and stopped training at epoch 17) | 
| GPUs           | 1 locally on CPU, and 4 on floyd |
| Batch Size     | 64 for final model locally |
| Learning Rate  | Started at 0.001, and it was still this at epoch 3 for the final model. |
| Test Set Size  | 20% of downsampled dataset.   |
| Dropout Percentage | 25% |
| Side Adjustment | 0.25 |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, on my lightweight dataset on my laptop, I included 3 laps each of center lane driving from both the clockwise and counterclockwise directions. NOTE: I used the keyboard arrows, as I do not have a gaming controller that will work with my laptop. 

![alt text][center_lane_simple]

On my larger dataset on floyd, I also included two laps each of as centered of center lane driving that I could manage on the complex track, in both directions. 

![alt text][center_lane_complex]

From there, I tested the model to see where it was having difficulty on the track. It was having a lot of trouble with recovering from the red line, so I added in an example of recovery from that type of line back to center. These images show what a recovery looks like starting from the edge and ending up back in the middle:

![alt text][red_line_recover_1]
![alt text][red_line_recover_2]
![alt text][red_line_recover_3]

I also perennially had issues with the car missing the first corner after the bridge and going onto the dirt track, so I included three examples of taking that corner successfully. 

![alt text][dirt_track_1]
![alt text][dirt_track_2]
![alt text][dirt_track_3]

While that was concerning for leaking information and potential for overfitting, and as such, in the future, I'd probably plan out different types of recoveries and make sure I can keep those proportional to the full lap center lane driving classes and use a track for simulation that was of similar difficulty that the model had never seen before. 

To augment the data sat, I did several things, all implemented from ideas posed in Yadav's blog post cited earlier. Initially, I tried using the built-in Keras ImageGenerator to augment the image, but had better results when I made my own custom function and added in some augementations not available in the Keras package. 

* Adjusting measurement of side images: This transformed images from the two sides into use as recovery images. The images did not change, but the measurements were offset by +/- 0.25 as discussed earlier in the solution design approach section. 

* Flipping the image from left to right, to boost clockwise and counterclockwise viewpoints:

![alt text][red_line_recover_1]
![alt text][flipped]

* Randomly changing the image brightness/darkness, to allow the model to generalize during different light levels and times of day. 

![alt text][center_lane_simple]
![alt text][brightness]

* Translating the image to offset a little bit in realistic ways while driving. This didn't change the image much visually, but did change it slightly.

![alt text][center_lane_simple]
![alt text][translation]

* Randomly adding a shadow onto the image, to help the model learn that it didn't have to avoid shadows. 
![alt text][center_lane_simple]
![alt text][shadow]

After the collection process, I had 9024 number of data points on my lightweight local dataset, and 18014 data points on my larger floyd dataset. These were opened up as color images in the same color space, and were normalized and cropped within the model pipeline itself so this could happen on the GPU. 

However, on my local dataset, the average measurement was -0.01055, and on the floyd dataset, it was at -0.00271, with the total of images between the ranges of -.1 and +.1 much larger than the total of the other images. So, I used random undersampling from the imbalanced-learn (imblearn) package to randomly downsample the record within that range and reduce those records by 10%.

After downsampling, I randomly shuffled the data set and put 20% of the downsampled data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs ended up being 3 as this facilitated a model that actually stayed on the road, but it looks like while the ideal number of epochs was typically relatively small, it could vary quite a bit depending on model setup and image augmentation. 

In all, this was a really fun but really challenging project! I typically use recurrent neural networks for text analytics, so this was a really interesting change of pace :-) 
