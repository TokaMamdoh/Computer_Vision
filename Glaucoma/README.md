# Glaucoma
![](https://github.com/TokaMamdoh/Computer_Vision/blob/3b044cd6f987060fdcf4d3497ad269fdc5c833a9/Glaucoma/images/eye.png)

## Abstract
 There is no doubt that, the glaucoma is one of the most common causes of lifelong blindness among people. It is a disorder that causes gradual damage to the optic nerve over a period of time, resulting in a partial loss of vision. Glaucoma is a disease that affects most people all around the world, and it is one of the leading causes of blindness among people over 60 years of age. In order to avoid permanent vision loss as a result of this disease, early detection is crucial. Fundus imaging using a retinal camera is the most common method of screening for glaucoma, and it is widely used in the detection of this disease. As part of this study, we would be using the fundus images from the retina in order to train the system using the fundus images in order to improve the performance of the system. We will use the Convolution Neural Network Technique in order to predict the possibility of occurrence of Glaucoma based on this prediction.

## Methodology
The methodology of this work consists of firstly, choosing dataset and preform multiple preprocessing methods, secondly, build CNN model to predict glaucoma, finally,
evaluate the model.

### A. Glaucoma Dataset
A Chosen dataset is consisting of fundus images from Kaggle. The size of this dataset is 4854 images. This data has been separated to train data, validation data and test data, and each set contains two classes which is class0 represented a normal eye and class1 represented an eye with glaucoma. 

Color Fundus Retinal Photography Color Fundus Retinal Photography is done using a microscope attached flashed camera to capture color images of the interior surface of the eye, including the retina, retinal vasculature, optic disc, macula, and posterior pole.

![](https://github.com/TokaMamdoh/Computer_Vision/blob/3b044cd6f987060fdcf4d3497ad269fdc5c833a9/Glaucoma/images/Normal%20images.PNG)   
Normal image

![](https://github.com/TokaMamdoh/Computer_Vision/blob/3b044cd6f987060fdcf4d3497ad269fdc5c833a9/Glaucoma/images/Galucoma%20image.PNG)

Galucoma image

### B. Data manipulation
#### 1) Data preprocessing
Image Preprocessing enhance the features of images, remove noises from funds images. Pre-processing an image is comprised of two types: linear and nonlinear. In the linear method, the algorithm can be used to apply linearto all pixels without defining the image as corrupted or uncorrupted before applying linear. With the nonlinear method, the algorithm can only apply pixels by defining which pixels are corrupted and which pixels are not Fig.6: Normal retina image Fig.7: Glaucoma image corrupted in the first place. In addition to this, the corrupted image is then filtered by the specific algorithm, and the uncorrupted image is retained. The results of nonlinear filters are better than those of linear filters when compared with them.


In this step we applied several preprocessing methods including:
#### 1- Convert images to Grayscale
The use of grayscale simplifies the algorithm and reduces the computational requirements compared to other algorithms that use color. beside from presenting the image in grayscale, it also helps in reducing noise due to the fact that color information doesn't readily assist the algorithm in identifying all the important edges (changes in pixel value) or other features in the image that might be important to identify. We apply grayscale to simplifies the algorithm and reduces computational requirements.
#### 2- Remove noise
When it comes to removing or reducing the noise from an image, noise removal algorithms can be regarded as one of the most important steps. It is the objective of noise removal algorithms to reduce or eliminate the visibility of noise by smoothing the entire image thereby leaving areas close to contrast boundaries, but not removing them. We picked Gaussian from open2Cv as it works on handling high and low frequency image beside it has the properties of having no overshoot to a step function input while minimizing the rise and fall time. In terms of image processing, any sharp edges in images are smoothed while minimizing too much blurring.
#### 3- Resize images
When resizing an image, it’s important to keep in mind the aspect ratio — which is the ratio of an image’s width to its height. Ignoring the aspect ratio can lead to resized images that look compressed and distorted: by following the below equation

                                                  (Pic - np.min(pic)) / (np.max(pic) - np.min(pic))

We managed to set all images in the dataset to the same size and still save our image ratio to meet OpenCV's requirements.

![](https://github.com/TokaMamdoh/Computer_Vision/blob/3b044cd6f987060fdcf4d3497ad269fdc5c833a9/Glaucoma/images/after-perprocessing.PNG)

Image after perprocessing 


#### 2) Data Augmentation
By applying data augmentation on the fundus images, it helps to expand the size of train and validation sets by creating modified data from the existing one by rotation, rescaling, flipping, shifting, sharing, zooming, brightness, and normalization. To make the model learn on all the sides of the images to make it easy to it while predicting that help to avoid the overfitting.

### C. Modeling
Convolution Neural Network (CNN) used in image recognition and processing that is specifically designed to process pixel data. The main advantage of CNN compared to its predecessors is that it automatically detects the important features without any human supervision. It learns distinctive features for each class by itself. CNN is also computationally efficient. The reason to select CNN model that it can overcome some problem in some machine learning algorithms (logistic regression, random forest, SVM) that require proper features to perform the classification that may fail to classify properly and the classifier accuracy will be lower. Second, when the required features and the appropriate classification model are selected to improve the training, they take a lot of time. So, CNN the best choice to classify this
fundus images.

![](https://github.com/TokaMamdoh/Computer_Vision/blob/3b044cd6f987060fdcf4d3497ad269fdc5c833a9/Glaucoma/images/CNN%20architecture.PNG)

CNN architecture

By applying the CNN Model with Sequential Method to the Glaucoma dataset. Overall idea is to create an array of layers and pass it to keras.Sequential method.[3] It enabled the model to predict if the image is affected by Glaucoma or Normal eye by passing the data through four convolution layers and flatting the results to feed into a DNN then visualization of model architecture by .summary() method. and fitting the model by specific number of epochs and comparing the training accuracy with validation accuracy then with testing in evaluation. The overall idea is to pass a layer to the next layer as a functional input. 

## Performance evaluation
In order to select the best technique to handle unstructured data that is recognized through images. We prefer to go with deep learning algorithm specially CNN that has high performance with images that can be represented as a matrix with pixel values. The reason to select CNN model that it can overcome some problem in some machine learning algorithms (logistic regression, random forest, SVM) that require that require proper features to perform the classification that may fail to classify properly and the classifier accuracy will be classification model are selected to improve the training, they take a lot of time.  Some operation such as pooling, flatting and regularization technique such as Dropout are used to improve the performance of CNN model that achieved in train accuracy by 84.88%, validation accuracy by 81.32%, and test accuracy by 88.18%. This graph shows that the rate of train accuracy change well after 25 epochs and the rate of training and validation are well, so low bias and low variance occur. When bias and variance are low, the CNN model may consistent and accurate.

![](https://github.com/TokaMamdoh/Computer_Vision/blob/3b044cd6f987060fdcf4d3497ad269fdc5c833a9/Glaucoma/images/model%20accuracy.PNG)

Model accuracy

## Graphical User Interface (GUI)
A graphical user interface (GUI) is a type of user interface through which users interact with electronic devices via visual indicator representations. Our website aims to help doctors and end user to use our tool in smoothy and immediately get check result with couple of clicks, and also enable us to visualize our glaucoma detection model.

![](https://github.com/TokaMamdoh/Computer_Vision/blob/3b044cd6f987060fdcf4d3497ad269fdc5c833a9/Glaucoma/images/GUI.PNG)

## Conclusion
In conclusion, let's summarize what we have gone through. First, we started with selecting an idea, and we sited on Detecting Glaucoma with the CNN algorithm due to an increasing number of cases that get blind because of this silent thief. As a next step, we looked for a dataset that could satisfy our needs, and the chosen data was 4854 images, split into three sets: training set, validation set, and testing set, and each set contains two classes. The next phase was applying data preprocessing to remove data noise and other things that can negatively affect our data quality. Hence, we used algorithms like Gaussian and grayscale beside resizing images to overcome confusing image ratios. Still, we faced problems with dataset size as we know a bigger dataset size leads to better-trained models and better prediction, so we applied data augmentation and the other mentioned techniques to increase our dataset and avoid overfitting. In the modeling phase, we used the CNN algorithm to build a deep neural network with four convolution layers to predict if the image has Glaucoma or not. We achieved this model with 88% accuracy. Now we keep looking at enhancing our model to achieve higher accuracy. Besides, we managed to build a website to make it more easy and fixable for doctors and end users to use our tool. 

