# LeNet_CNN
## CIFAR-10
A dataset created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. 
It consists of 60000 images with a 32x32 resolution. The dataset is split into
50000 pictures for the training set and 10000 images for testing purposes.
The images consist of 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, 
Frog, Horse, Ship, and a Truck. The image below shows each of the classes and
the a few examples of each class.

<img width="476" alt="Cifar-example" src="https://github.com/Pranav2328/LeNet_CNN/assets/85324957/327f552a-a945-471b-84e7-546b5f5e14af">

## Neural Network Structure
The Neural network I built is a convolutional neural network based on the architectectural design from [Yann LeCun paper.](http://yann.lecun.com/exdb/lenet/) I used the Tensorflow and Keras library to implement the model. Apart from the convolutional layers I normalized the data and created fake data using the ImageDataGenerator library in order to improve the accuracy of the model.

## Model Analysis
The graph below shows the Accuracy of the model for the first 10 epochs.

<img width="615" alt="Screenshot 2023-08-11 at 11 38 19 AM" src="https://github.com/Pranav2328/LeNet_CNN/assets/85324957/0208d696-9c51-40e3-baff-0f349ed5d769">

The graph below shows the Loss of the model after the first 10 epochs.

<img width="615" alt="Screenshot 2023-08-11 at 11 38 28 AM" src="https://github.com/Pranav2328/LeNet_CNN/assets/85324957/7737bd34-5261-4547-a645-1ad6fee39ab2">

## Model Evaluation
Upon training the model for different number of epochs the accuracy of the model
on the training set increases drastically with more number of epochs up to about
90% buut when the model attempts to classify the test set it only succeeds at a 
60% rate. 
