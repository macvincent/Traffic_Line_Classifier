# **Traffic Sign Recognition** 

The goals / steps of this project were the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./distribution_images/bar1.png "train"
[image2]: ./distribution_images/bar2.png "test"
[image3]: ./distribution_images/bar3.png "valid"
[image4]: ./test_images/30mph.jpg "30mph"
[image6]: ./test_images/general_caution2.png "General Caution 2"
[image7]: ./test_images/keep_right.jpg "Keep Right"
[image8]: ./test_images/stop.jpg "Stop"
[image9]: ./test_images/bumpy_road.jpg "Bumpy Road"
[image10]: ./test_images/images.jpg "Images"
[image11]: ./test_images/curve.png "Curve"

## Rubric Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 42

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the datasets. There are bar charts the number of examples of each class of the traffic sign contained in each dataset.

#### Training Dataset

![alt text][image1]

#### Validation Dataset
![alt text][image3]

#### Validation Dataset
![alt text][image2]

Here we see that certain classes have more test examples than the others in each data set.

### Design and Test a Model Architecture

In the `normalise` function I normalized the pixel values to be decimals between 0 and 1, as this generally works better with networks.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					                | 
|:---------------------:|:-------------------------------------------------------------:| 
| Conv2D      	        | 3x3 stride, same padding, 32 outputs filters, 32*32*3 input 	|
| MaxPool2D      	    | 2x2 stride and pool size				                        |
| Dropout       	    | 0.25 rate                 									|
| Conv2D      	        | 3x3 stride, same padding, 64 outputs filters 	                |
| MaxPool2D      	    | 2x2 stride and pool size				                        |
| Dropout       	    | 0.25 rate                 									|
| Conv2D      	        | 3x3 stride, same padding, 128 outputs filters                 |
| MaxPool2D      	    | 2x2 stride and pool size				                        |
| Dropout       	    | 0.25 rate                 									|
| Flatten       		| Make input one dimensional        		    				|
| Dense 				| Fully connected dense layer           						|
| Dropout       	    | 0.25 rate                 									|
| Dense            		|  `nclasses` output filter with softmax activation				|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Using the architecture descibed above, I passed the entire training and validation set over 7 epochs. To avoid overfitting, I chose the number of epochs after considering the accuracy of the training and validation sets after each epochs.

My final model results were:
* training set accuracy of 0.9621
* validation set accuracy of 0.9612
* test set accuracy of 0.9447

I made use of a tensorflow example architecture I had previously used in classifying a [dog vs cat dataset](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c01_dogs_vs_cats_without_augmentation.ipynb#scrollTo=wqtiIPRbG4FA). However, I had to change the input shape in the first layer, change the number of classes in the softmax layer, and add dropout layers within the network to ensure the power of individual nodes are maximized. I chose this architecture because it has to do with classifying colored images and, with a 0.9313 accuracy on out validation set afer 5 Epochs, it clearly works well with our dataset.
 

### Test a Model on New Images
I tried our images on images not included in our test data set. Here are some German traffic signs that I found on the web:

![alt text][image4] ![alt text][image6] 
![alt text][image7] ![alt text][image8]
![alt text][image9]![alt text][image10]
![alt text][image11]

Here are the results of the prediction:

| Image			             |     Prediction	        					| 
|:--------------------------:|:--------------------------------------------:| 
| Speed limit (30km/h) 	     | Speed limit (70km/h)  						|
| General caution            | General caution                              |
| Keep right			     | Keep right          							|
| Stop                       | Stop                                         |
| Bumpy road    		     | Bumpy road									|
| Speed limit (70km/h)	     | Speed limit (70km/h)			 				|
| Dangerous curve to the left| Dangerous curve to the left                  |

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 0.8571. This compares favorably with the accuracy on the test set which is 0.9447. For the first speed limit image, our model can detect that it is a speed limit sign, but it fails to correctly detect the speed written on it, which shows that the value of the speed limit detected by our model cannot be trusted.

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

For the first image, the model is really sure that it is a Speed limit sign (probability of 0.99999940). While the image does contain a Speed limit sign, the model fails to read the value of the speed limit. The top five soft max probabilities were

| Probability         	|     Prediction	  					| 
|:---------------------:|:-------------------------------------:| 
| 9.9999940e-01       			| Speed limit (70km/h)   			    | 
| 6.4220234e-07    				| Speed limit (20km/h)					|
| 0.0000000e+00					| Speed limit (30km/h)					|
| 0.0000000e+00	      			| Speed limit (50km/h)					|
| 0.0000000e+00				    | Speed limit (60km/h)      		    |

For the second image (Correct)
| Probability         	|     Prediction       					| 
|:---------------------:|:-------------------------------------:| 
| 1.       			    | Bumpy road          	            	| 
| 0.    				| Speed limit (20km/h)					|
| 0.					| Speed limit (30km/h)					|
| 0.	      			| Speed limit (50km/h)					|
| 0.				    | Speed limit (60km/h)      			|

For the third image (Correct)
| Probability         	|     Prediction	  					| 
|:---------------------:|:-------------------------------------:| 
| 1.       			    | Speed limit (70km/h)          		| 
| 0.    				| Speed limit (20km/h)					|
| 0.					| Speed limit (30km/h)					|
| 0.	      			| Speed limit (50km/h)					|
| 0.				    | Speed limit (60km/h)      			|

For the fourth image (Correct)
| Probability         	|     Prediction	  					| 
|:---------------------:|:-------------------------------------:| 
| 1.       		    	| Keep right         	            	| 
| 0.    				| Speed limit (20km/h)					|
| 0.					| Speed limit (30km/h)					|
| 0.	      			| Speed limit (50km/h)					|
| 0.				    | Speed limit (60km/h)      			|

For the fifth image (Correct)
| Probability         	|     Prediction	  					| 
|:---------------------:|:-------------------------------------:| 
| 1.       			    | Stop          		                | 
| 0.    				| Speed limit (20km/h)					|
| 0.					| Speed limit (30km/h)					|
| 0.	      			| Speed limit (50km/h)					|
| 0.				    | Speed limit (60km/h)      			|

For the sixth image (Correct)
| Probability         	|     Prediction	  					| 
|:---------------------:|:-------------------------------------:| 
| 1.       		    	| Dangerous curve to the left     		| 
| 0.    				| Speed limit (20km/h)					|
| 0.					| Speed limit (30km/h)					|
| 0.	      			| Speed limit (50km/h)					|
| 0.				    | Speed limit (60km/h)      			|

For the seventh image (Correct)
| Probability         	|     Prediction	  					| 
|:---------------------:|:-------------------------------------:| 
| 1.       			    | General caution                  		| 
| 0.    				| Speed limit (20km/h)					|
| 0.					| Speed limit (30km/h)					|
| 0.	      			| Speed limit (50km/h)					|
| 0.				    | Speed limit (60km/h)      			|