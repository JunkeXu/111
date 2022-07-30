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

![1.png](attachment:1.png)


## Packages

### Install the tensorflow and keras

In the newly created virtual environment tf1, install the two libraries using the following command.

pip install tensorflow==1.14.0

![1659191227%281%29.png](attachment:1659191227%281%29.png)


pip install keras==2.2.5
![1659191294%281%29.jpg](attachment:1659191294%281%29.jpg)

### Installing the nb_conda_kernels package

conda install nb_conda_kernels

![1659192157%281%29.png](attachment:1659192157%281%29.png)

### Install version 1.16.4 of numpy

pip install numpy==1.16.4

![image.png](attachment:image.png)

### Installing the pillow and matplotlib

pip install pillow
![1659210198%281%29.png](attachment:1659210198%281%29.png)


pip install matplotlib

![1659210242.png](attachment:1659210242.png)

### Open the Jupyter Notebook (tf1) environment.

![image.png](attachment:image.png)

Click [New] → [Python [in tf1 environment]] to create the python file.

![image.png](attachment:image.png)

## Dog and cat data classification modelling

Once the dataset has been downloaded, unzip it as follows.

![image.png](attachment:image.png)

The code for classifying images of dogs and cats is as follows:

```
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
```


The classification of the dog and cat images is shown in the following figure.

![1659193037%281%29.png](attachment:1659193037%281%29.png)

![1659194393%281%29.png](attachment:1659194393%281%29.png)


