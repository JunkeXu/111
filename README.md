# Image Recognition Dogs vs Cats

## Basic Overview
Using cat and dog dataset to train a convolutional neural network model and achieve an accuracy of over 90% for cat and dog recognition

##  Installation
1. Install Python: https://www.python.org/downloads/release/python-387/
2. Install Anaconda: https://www.anaconda.com/products/distribution

## Data

Image data from the kaggle website: https://www.kaggle.com/c/dogs-vs-cats/data

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

### Open the Jupyter Notebook (tf1) environment.

![9.png](https://github.com/JunkeXu/111/blob/main/figure/9.png)

Click [New] → [Python [in tf1 environment]] to create the python file.

![10.png](https://github.com/JunkeXu/111/blob/main/figure/10.png)

## Dog and cat data classification modelling

Once the dataset has been downloaded, unzip it as follows.

![11.png](https://github.com/JunkeXu/111/blob/main/figure/11.png)

The code for classifying images of dogs and cats is as follows:

'
# The path where the original directory is located
original_dataset_dir = 'C:\\Users\\xjk\\Desktop\\Cat_And_Dog\\train\\'

# Catalogues after data set classification
base_dir = 'C:\\Users\\xjk\\Desktop\\Cat_And_Dog\\train1'
os.mkdir(base_dir)
#Catalogue of training, validation and test datasets
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
# Cat training pictures in the catalogue
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
# Dog training pictures in the catalogue
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
# Cat verification of the directory where the image is located
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
# Directory where the dog validation dataset is located
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
# The directory where the cat test dataset is located
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
# Directory where the dog test dataset is located
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)
# Copy the first 5000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(5000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
# Copy the next 2500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(5000, 7500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst) 
# Copy the next 2500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(7500, 10000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
# Copy the first 5000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(5000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
# Copy the next 2500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(5000, 7500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
# Copy the next 2500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(7500, 10000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
'


The classification of the dog and cat images is shown in the following figure.

![12.png](https://github.com/JunkeXu/111/blob/main/figure/12.png)

![13.png](https://github.com/JunkeXu/111/blob/main/figure/13.png)

![14.png](https://github.com/JunkeXu/111/blob/main/figure/14.png)

## Summary of Analysis

When comparing the baseline model, it is clear that the overall trend in loss is smaller for the model obtained with image augmentation and for the model obtained with image augmentation and the addition of a dropout layer. The addition of a dropout layer under the influence of data augmentation allows the training curve to follow the validation curve more closely and the fluctuations to be reduced, resulting in better training results, but with a slight decrease in accuracy. In addition, when we go through the feature extraction, we feed the image into VGG19's convolutional layer and let it extract the features from the image directly for us, we do not train and change VGG19's convolutional layer with our own images, the parameter tuning approach is that we train the convolutional layer provided by VGG19 with our own data to a limited extent, so that it can learn relevant information from our images. from our images. Using the VGG19 model, we can see that the network trained on more than a million images is much better than the one we trained, and the network verifies the images correctly at more than 99%.

## Conclusion

