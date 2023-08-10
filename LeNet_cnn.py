import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import numpy as np

def build_lenet():
    lenet = models.Sequential()

    #Convolutional layers
    lenet.add(layers.Conv2D(filters=64, kernel_size=(9,9), activation='relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
    lenet.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    lenet.add(layers.Conv2D(filters=32, kernel_size=(9,9), activation='relu', kernel_initializer='he_normal'))
    lenet.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    #Dense layers
    lenet.add(layers.Flatten())
    lenet.add(layers.Dense(units=120, activation='relu', kernel_initializer='he_normal'))
    lenet.add(layers.Dense(units=60, activation='relu', kernel_initializer='he_normal'))
    lenet.add(layers.Dense(units=10, activation='softmax', kernel_initializer='he_normal'))

    lenet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return lenet
    
if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Data preprocessing 
    # X_train /= 255.0
    # X_test /= 255.0
    mean  = [125.307, 122.95, 113.865]
    std   = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        X_train[:,:,:,i] = (X_train[:,:,:,i] - mean[i]) / std[i]
        X_test[:,:,:,i] = (X_test[:,:,:,i] - mean[i]) / std[i]

    # using real-time data augmentation
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen.fit(X_train)

    # Define per-fold score containers
    # acc_per_fold = []
    # loss_per_fold = []

    # Merge inputs and targets
    # inputs = np.concatenate((X_train, X_test), axis=0)
    # targets = np.concatenate((y_train, y_test), axis=0)

    #Create cross validation set
    # X_train, X_, y_train, y_ = train_test_split(inputs, targets, test_size=0.40, random_state=1)
    # X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.50, random_state=1)
    # kfold = KFold(n_splits=5, shuffle=True)

    # fold_no = 1
    # for train, test in kfold.split(inputs, targets):

    #Create LeNet Model
    model = build_lenet()

        # # Generate a print
        # print('------------------------------------------------------------------------')
        # print(f'Training for fold {fold_no} ...')

    #Train LeNet model
    model.fit(datagen.flow(X_train, y_train, batch_size=128), epochs=10, shuffle=True)
    print(model.evaluate(X_test, y_test))

        # # Generate generalization metrics
        # scores = model.evaluate(inputs[test], targets[test], verbose=0)
        # print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        # acc_per_fold.append(scores[1] * 100)
        # loss_per_fold.append(scores[0])

        # # Increase fold number
        # fold_no = fold_no + 1