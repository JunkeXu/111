# Image Recognition Dogs vs Cats

[![forthebadge](http://forthebadge.com/images/badges/made-with-python.svg)](http://forthebadge.com)

Table of Contents
=================

  * [Basic Overview](#Basic-Overview)
  * [Installation](#Installation)
  * [Data](#Data)
  * [Managing virtual environments](#Managing-virtual-environments)
    * [Create own virtual environment](#Create-own-virtual-environment)
    * [Activating the environment](#Activating-the-environment)
  * [Packages](#Packages)
  * [Open environment](#Open-the-Jupyter-Notebook-(tf1)-environment)
  * [Dog and cat data classification modelling](#Dog-and-cat-data-classification-modelling)
  * [Summary of Analysis](#Summary-of-Analysis)
  * [Conclution](#Conclution)
  * [Results](#Results)
  * [Run time](#Run-time)
  * [Author](#Author)

## Basic Overview

![20.png](https://github.com/JunkeXu/111/blob/main/figure/20.png)

Using cat and dog dataset to train a convolutional neural network model and achieve an accuracy of over 90% for cat and dog recognition

##  Installation
1. Install Python: https://www.python.org/downloads/release/python-387/
2. Install Anaconda: https://www.anaconda.com/products/distribution

## Data

Data set from a competition on kaggle：[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)，The training set has 25,000 sheets, split 50/50 between dogs and cats. The test set of 12,500 sheets is not labelled as cat or dog.

## Managing virtual environments

Next we can use anaconda to create our individual python environments. The examples below are all done from the command line, so open command line.

### Create own virtual environment

Open the cmd command terminal and create a virtual environment.

conda create -n tf1 python=3.6

![1.jpg](https://github.com/JunkeXu/111/blob/main/figure/1.png)


### Activating the environment

activate

conda activate tf1

![2.png](https://github.com/JunkeXu/111/blob/main/figure/2.png)


## Packages

In the newly created virtual environment tf1, install libraries using the following command.

pip install tensorflow==1.14.0

![3.png](https://github.com/JunkeXu/111/blob/main/figure/3.png)


pip install keras==2.2.5

![4.png](https://github.com/JunkeXu/111/blob/main/figure/4.png)

pip install numpy==1.16.4

![6.png](https://github.com/JunkeXu/111/blob/main/figure/6.png)

conda install nb_conda_kernels

![5.png](https://github.com/JunkeXu/111/blob/main/figure/5.png)

pip install pillow

![7.png](https://github.com/JunkeXu/111/blob/main/figure/7.png)

pip install matplotlib

![8.png](https://github.com/JunkeXu/111/blob/main/figure/8.png)

pip install pandas

![22.png](https://github.com/JunkeXu/111/blob/main/figure/22.png)

## Open the Jupyter Notebook (tf1) environment

![9.png](https://github.com/JunkeXu/111/blob/main/figure/9.png)

Click [New] → [Python [in tf1 environment]] to create the python file.

![10.png](https://github.com/JunkeXu/111/blob/main/figure/10.png)

## Dog and cat data classification modelling

Once the dataset has been downloaded, unzip it as follows.

![11.png](https://github.com/JunkeXu/111/blob/main/figure/11.png)

The code for classifying images of dogs and cats is as follows:

![19.png](https://github.com/JunkeXu/111/blob/main/figure/19.png)

The classification of the dog and cat images is shown in the following figure.

![12.png](https://github.com/JunkeXu/111/blob/main/figure/12.png)

![13.png](https://github.com/JunkeXu/111/blob/main/figure/13.png)

![14.png](https://github.com/JunkeXu/111/blob/main/figure/14.png)

## Summary of Analysis

When comparing the baseline model, it is clear that the overall trend in loss is smaller for the model obtained with image augmentation and for the model obtained with image augmentation and the addition of a dropout layer. The addition of a dropout layer under the influence of data augmentation allows the training curve to follow the validation curve more closely and the fluctuations to be reduced, resulting in better training results, but with a slight decrease in accuracy. In addition, when we go through the feature extraction, we feed the image into VGG19's convolutional layer and let it extract the features from the image directly for us, we do not train and change VGG19's convolutional layer with our own images, the parameter tuning approach is that we train the convolutional layer provided by VGG19 with our own data to a limited extent, so that it can learn relevant information from our images. from our images. Using the VGG19 model, we can see that the network trained on more than a million images is much better than the one we trained, and the network verifies the images correctly at more than 99%.

## Conclusion

![15.png](https://github.com/JunkeXu/111/blob/main/figure/15.jpg)

The trained model is not very accurate and there is an increasing trend of loss, which may lead to overfitting.

![16.png](https://github.com/JunkeXu/111/blob/main/figure/16.png)

The overall trend in loss is to become smaller, with Volatility on the high side during training.

![17.png](https://github.com/JunkeXu/111/blob/main/figure/17.png)

The overall trend in loss is to become smaller，Volatility has also been reduced, but accuracy is low.

![18.png](https://github.com/JunkeXu/111/blob/main/figure/18.png)

The training was perfect, but with the help of the VGG19 model.

![21.jpg](https://github.com/JunkeXu/111/blob/main/figure/21.jpg)

The training was perfect and was generated by parameter tuning on the VGG19 model.

## Results

The result can be seen in [project.csv](https://github.com/JunkeXu/111/blob/main/project.csv).

## Run time

The approximate time to run the entire notebook is around 2 hours.

## Author

Junke Xu 

email: junke.xu@ucdconnect.ie
