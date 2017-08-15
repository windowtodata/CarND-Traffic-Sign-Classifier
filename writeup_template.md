#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./submission/plot1.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./samples/childrenxing.jpg "Traffic Sign 1"
[image5]: ./samples/Do-Not-Enter.jpg "Traffic Sign 2"
[image6]: ./samples/road_work.jpg "Traffic Sign 3"
[image7]: ./samples/speed30.jpg "Traffic Sign 4"
[image8]: ./samples/speed60.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the frequency distribution of the classes. The classes are not uniformly distributed as expected. We can use this information to make the class distribution uniform(balanced) by simnply augmenting the classes that are less in numbers. However, since we will be runnning sufficient number of epochs, that will not be necessary.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color provides minimal information for classifying the images, and may lead to noise. The images are fairly distinguishable in grayscale. This also reduces data for training.

Although I did not save the grayscale images (I normalize in the same loop), here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it helps the neural network to converge faster if we normalize.
 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   		     		| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs  28x28x6 	|
| RELU					| Applying non-linearity   						|
| Max pooling 2x2      	| 2x2 stride,  valid-padding, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, valid-padding. Outputs 10x10x16  	|
| RELU 					| Applying non-linearity						|
| Max pooling 2x2   	| 2x2 stride,  valid-padding, Outputs 5x5x16 	|
| Flatten				| Outputs 400 									|
| Fully connected		| Outputs 120									|
| RELU 					| Applying non-linearity						|
| Fully connected		| Outputs 84 									|
| RELU 					| Applying non-linearity						|
| Softmax				| Outputs 43   									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the LeNet architecture. I added a drop-out before the last fully-connected layer for regularization.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? 1.0
* validation set accuracy of ? 0.932
* test set accuracy of ? 0.926

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen? 
* Why did you believe it would be relevant to the traffic sign application? 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it also contains writing below the actual sign. Also the forth image has a distracting background (trees).

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 28,Children crossing 	| 28,Children crossing  						| 
| 17,No entry			| 17,No entry   								|
| 25,Road work      	| 25,Road work  								|
| 1,Speed limit (30km/h)| 1,Speed limit (30km/h)            			|
| 3,Speed limit (60km/h)| 3,Speed limit (60km/h)			   			|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.6%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 120th cell of the Ipython notebook.
[28, 29, 30,  3, 23],
[17, 34,  9, 28,  3],
[25, 21, 30, 31, 20],
[ 1,  5,  0,  3,  4],
[ 3,  2,  1, 36, 38]

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain children crossing. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .935         			| 28,Children crossing  						| 
| .041     				| 29,Bicycles crossing							|
| .0139					| 30,Beware of ice/snow							|
| .0046	      			| 3,Speed limit (60km/h)		 				|
| .0029				    | 23,Slippery road     							|





### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


