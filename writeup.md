# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./examples/visualization_augmented.png "Visualization Augmented"
---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histigram showing the distribution of data per class.

![alt text][image1]

As you can see the training data is heavily skewed towards a few specific classes. We can improve our accuracy by augmenting our data by performing rotation , translation and brightness changes from samples in the dataset for classes with a lower number of examples. After augmentation, the distribution is:

![alt text][image9]



Here is an example of an original image and an augmented image:

![alt text][image3]



### Design and Test a Model Architecture

1. As a first step, I decided to convert the images to grayscale because the the sign should be identifiable with the greyscale image and becasue it is informationally less dense.
2. For making it greyscale my first though t was to give more weight to red as that is an important color, however the validation of the model proved that such an approach is not useful.
3. I normalised the image because it makes sense to consider all pixels of all images to be between zero and one, as that will leed to uniformity.

Here is an example of a traffic sign image before and after normalisation.

![alt text][image2]


The difference between the original data set and the augmented data set is the following ... 


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x6    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5x16	| 1x1 stride, valid padding, outputs 10x10x16 	|
| Max pooling 2x2	    | 2x2 stride,  outputs 14x14x6 				    |
| Flatten & concatanate | uses max pooling layer outputs 1576			|
| Fully connected		| ouputs 43 logits								|
 

These are the Hyper Parameters that I used:

Epochs=20
BATCH_SIZE=128
rate=0.0009

For training the dropout rate was set to 50%.

After settling on the final model, I increased the Epochs to 60.

To train the model, I used an Adam optimiser as it is said to be affective for LeNet. The loss function was the resuced mean of the softmax cross entropy function.

I used an iterative approach:
1. using normalise dot (Prioritising red during greyscaling) : Validation Accuracy = 0.908

2. using normalise sum for grayscale : Validation Accuracy =0.898

3. using augmented data :  Validation Accuracy = 0.895

4. changes to data augmentation and dropout, increased epochs to 20 from 10 and reduced learning rate to 0.0009
Training started : Validation Accuracy = 0.951

5. using LeNet 2 and reverting back to normalise sum: Validation Accuracy = 0.955

### Testing accuracy for final model

My final model results were:
* validation set accuracy of = 0.959 
* test set accuracy of = 0.936


### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]



Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Right      		| Keep Right   									| 
| Turn left ahead   	| Turn left ahead								|
| Road work				| Road work										|
| Priority road     	| Priority road 				 				|
| Speed limit (60km/h)	| Speed limit (60km/h) 							|
| Speed limit (30km/h)	| Speed limit (30km/h) 							|
| General caution   	| General caution   							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.
