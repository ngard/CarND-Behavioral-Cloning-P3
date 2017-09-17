
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./center.png "Center Lane Driving"
[image2]: ./recovery.png "Recovery Driving"
[image3]: ./specific.png "Recovery Driving on a Specific Corner"
[image4]: ./normal.jpg "Normal Image from Front Camera"
[image5]: ./flipped.jpg "Flipped Image from Front Camera"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizes the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is almost the same to the NVIDIA network (shown in the course) which is not so deep (means fewer learning time) but performs good enough for this project.
It consists of a convolution neural network with 5x5 or 3x3 filter sizes and depths between 24 and 64 (model.py lines 91-95) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 46).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 100).

At first, I introduced more than 1 layer, however, it turns out that multiple Dropout layers do not contribute to perform better (at least with this number of images or epochs). Therefore I limited to use the Dropout layer only at the last of the fully-connected layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 114-115). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track as can be seen in [video](./run2.mp4)

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center lane driving, recovering from the left and right sides of the road.

At first, I only used images of center lane driving, then the car did not learn to steer to keep center when the car deviate the center especially in curbs.
Then I added some images to teach how to keep center when the car is starting to deviate as the image below.

However, in some severe corners or corners without line, the car did not steer enough to keep center.
Therefore, I added another images to teach how to steer on those corners as the image below.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I firstly tried the simplest architecture which consists of only few fully-connected layers (model.py lines 48-51) and found that it works poor.

Then I tried a much deeper network based on VGG16 (model.py lines 53-88) to see if deeper network performs better.
However, I found that it works much poorer (steering fixed to a constant angle) even though it takes too much time to train the network.

Then I tried a NVIDIA network which is shown in the course and found that it performs good enough even though the model is relatively simple.

To combat the overfitting, I modified the model and added a Dropout layer at the tail.
I tried to introduce more Dropout layers between the fully-connected layers, however, the model does not work properly with them.

The final step was to run the simulator to see how well the car was driving around track one.
At first, the car did not know how to recenter when the car starts to deviate.
Then I taught how to do that and that enables the car to drive autonomously in most cases.

However, the car sometimes steer enough or strangely especially in specific places.
Therefore, I added anothe images to teach how to drive in those cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 44-107) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image  							|
| Cropping         		| 320x 65x3 RGB image  							| 
| Convolution 5x5     	| subsample = 2x2, outputs 158x 31x 24		 	|
| RELU					|												|
| Convolution 5x5     	| subsample = 2x2, outputs  77x 14x 36		 	|
| RELU					|												|
| Convolution 5x5     	| subsample = 2x2, outputs  37x  5x 48		 	|
| RELU					|												|
| Convolution 3x3     	| outputs  34x 3x 64						 	|
| RELU					|												|
| Convolution 3x3     	| outputs  32x 1x 64						 	|
| RELU					|												|
| Fully connected		| outputs  100 									|
| Fully connected		| outputs   50 									|
| Fully connected		| outputs   10 									|
| Dropout				| rate = 0.7									|
| Fully connected		| outputs    1 									|

The visualization of the architecture failed probably due to dependency problems in libraries which I failed to fix.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

However, with only images of center lane driving, the car does not learn how to recover to the center when the car is starting to deviate.
Therefore I recorded the vehicle recovering from the side of the road at some points of the course so that the vehicle would learn to recover when the car starts to deviate.
To take these images, I firstly parked the car diagonally, then set the steering to recover the vehicle to the center and started recording. I executed this process especially on the severe corners and the corners without lane markers where the car tends to deviate.

![alt text][image2]
![alt text][image3]

To augment the data sat, I also flipped images and angles because the model gets more accurate with the more data.
For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

After the careful collection process, I had around 10k number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Empirically, 3 was enough for the number of epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
