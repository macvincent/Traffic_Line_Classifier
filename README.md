# **Traffic Sign Recognition** 

The goals / steps of this project were the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./distribution_images/bar.png "distribution"
[image2]: ./distribution_images/loss.png "distribution"
[image3]: ./distribution_images/summary.png "distribution"
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
Here is a link to my [project code](./Traffic_Sign_Classifier.ipynb) implemented using a CNN and a link to a more accurate network designed using [transfer learning](./Transfer_Learning.ipynb) from the [MobileNet V1](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4) network.    

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the datasets. It shows the number of examples in each class for each dataset.

![alt text][image1]


Here we see that certain classes have more test examples than the others in each data set.

### Design Model Architecture

In the `normalise` function I normalized the pixel values to be decimals between 0 and 1, as this generally works better with networks.

For this projectm I made use of a Convoluted Neutral Network. My final model consisted of the 16 layers described below:

![summary of network structure][image3]


* Epochs: Using the architecture descibed above, I passed the entire training and validation set over 7 epochs. To avoid overfitting, I chose the number of epochs after considering the accuracy of the training and validation sets for each epochs. That is expressed with the graph described below:

![ghraph of training and validation loss per epoch][image2]

We see that there is no significant divergence between the loss per epoch of the validation and training sets.

* Optimizer: [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer) with a default learning rate of 0.001.

* Loss Function: [sparse_categorical_crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy).

* Batch Size: Default value of 32.

My final model results for the CNN were:
* training set accuracy of 0.9556
* validation set accuracy of 0.9637
* test set accuracy of 0.9495

The final results for the transfer learing model which made use of were:
* training set accuracy of 0.9907
* validation set accuracy of 0.9936
* test set accuracy of 0.9660

For the CNN model, I made use of a tensorflow example architecture I had previously used in classifying a [dog vs cat dataset](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c01_dogs_vs_cats_without_augmentation.ipynb#scrollTo=wqtiIPRbG4FA). However, I had to change the input shape in the first layer, change the number of classes in the softmax layer, and add dropout layers within the network to ensure the power of individual nodes are maximized and aovid overfitting. I chose this architecture because it was a Convolution Neural Network tuned for classifying colored images and, with a 0.9451 accuracy on out validation set afer 5 Epochs, it clearly works well with our dataset.
 

### Test a Model on New Images
I tried our model on images not included in the test data set. Here are some German traffic signs that I found on the web:

![alt text][image4] ![alt text][image6] 
![alt text][image7] ![alt text][image8]
![alt text][image9]![alt text][image10]
![alt text][image11]

Here are the results of the prediction:

| Image			             |     Prediction	        					| 
|:--------------------------:|:--------------------------------------------:| 
| Speed limit (30km/h) 	     | Speed limit (30km/h)  						|
| General caution            | General caution                              |
| Keep right			     | Keep right          							|
| Stop                       | Stop                                         |
| Bumpy road    		     | Bumpy road									|
| Speed limit (70km/h)	     | Speed limit (30km/h)			 				|
| Dangerous curve to the left| Dangerous curve to the left                  |

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 0.8571. This compares favorably with the accuracy on the test set which is 0.9495. For the second speed limit image, our model can detect that it is a speed limit sign, but it fails to correctly detect the speed written on it, which shows that the value of the speed limit detected by our model cannot be trusted.

The code for making predictions on the final model is located in the last code cell of the [Ipython notebook](./Traffic_Sign_Classifier.ipynb).

The top five softmax probabilities for our test images as predicted by the model are:

| Probability         	|     Prediction	  					| 
|:---------------------:|:-------------------------------------:| 
| 1.0000000e+00       			| Speed limit (30km/h)   			    | 
| 1.7098796e-37    				| Speed limit (50km/h)					|
| 0.0000000e+00					| Speed limit (20km/h)					|
| 0.0000000e+00	      			| Speed limit (60km/h)					|
| 0.0000000e+00				    | Speed limit (70km/h)      		    |

For the second image (Correct)
| Probability         	|     Prediction       					| 
|:---------------------:|:-------------------------------------:| 
| 1.       			    | Bumpy road          	            	| 
| 0.    				| Speed limit (20km/h)					|
| 0.					| Speed limit (30km/h)					|
| 0.	      			| Speed limit (50km/h)					|
| 0.				    | Speed limit (60km/h)      			|

For the third image (incorrect)
| Probability         	|     Prediction	  					| 
|:---------------------:|:-------------------------------------:| 
| 1.000000e+00   	    | Speed limit (30km/h)          		| 
| 7.807191e-09    		| Speed limit (20km/h)					|
| 0.000000e+00  		| Speed limit (50km/h)					|
| 0.000000e+00			| Speed limit (60km/h)					|
| 0.000000e+00  	    | Speed limit (70km/h)      			|

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

The transfer learning model was correct for all inputs.