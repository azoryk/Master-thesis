# -*- coding: utf-8 -*-
"""
This script trains the VGG-19 model on the small dataset "German POI 200"

The source of model was taken by the followning link: 
https://github.com/fchollet/deep-learning-models/blob/master/vgg19.py

Pretrained weights were downloaded by this link:
'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

@author: Andriy Zoryk
"""

# Import libraries
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.preprocessing.image import ImageDataGenerator


#%%

def vgg19_model(img_rows, img_cols, channel=1, num_classes=None):
    """
    VGG 19 Model for Keras
    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilZ2RVeVhKY0FyRmc/view?usp=sharing
    #Arguments
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
     # Returns
        A Keras model instance.
    """
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(channel, img_rows, img_cols)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    #the weights file should be downloaded from https://github.com/fchollet/deep-learning-models/releases
    model.load_weights('model_weights/vgg19_weights_tf_dim_ordering_tf_kernels.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))
    
    # Defines the metrics for calculating top3 accuracy
    from keras.metrics import top_k_categorical_accuracy
    def top_3_categorical_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3) 

    # Compiles the model with stochastic gradient descent
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', top_3_categorical_accuracy ])

    #Model overview
    model.summary()
    model.get_config()

    return model

#%%

"""
This method preprocess the images into an array by the needed size 
# Arguments
    img_rows, img_cols - resolution of inputs
    data_path - path to dataset
#Returns 
    X_train - train dataset
    X_test - test dataset 
    Y_train - train labels
    Y_test - test labels
    poi_list - list of labels

"""

def preprocessing(img_rows, img_cols, data_path):
    # Define data path 
    data_dir_list = os.listdir(data_path) #MacOS creates automatically '.DS_Store' file in each folder
    if data_dir_list[0] == '.DS_Store':
        data_dir_list = os.listdir(data_path)[1:]
    print (data_dir_list)
    
    #list of all the images
    img_data_list=[]

    #total number of images
    num_samples = 0


    #image preprocessing
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'\\'+ dataset)
        if img_list[0] == '.DS_Store':
            img_list = os.listdir(data_path+'\\'+ dataset)[1:] 
        num_samples +=len(img_list)
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        
        for img in img_list:
                input_img= imread(data_path + '\\'+ dataset + '\\'+ img, mode = img_mode)
                input_img_resize = imresize(input_img, (img_rows, img_cols))
                img_data_list.append(input_img_resize)
    
    print("The total number of images: " + str(num_samples))
    
    #array of all the images
    img_data = np.array(img_data_list, dtype = 'float32')
    print(img_data.shape)
    img_data = img_data.astype('float32')
    img_data /= 255
    print (img_data.shape)
    
    if num_channel==1:
    	if K.image_dim_ordering()=='th':
    		img_data= np.expand_dims(img_data, axis=1) 
    		print (img_data.shape)
    	else:
    		img_data= np.expand_dims(img_data, axis=4) 
    		print (img_data.shape)
    		
    else:
    	if K.image_dim_ordering()=='th':
    		img_data=np.rollaxis(img_data,3,1)
    		print (img_data.shape)
    
    # create the array of labels   
    label = np.ones((num_samples,), dtype=int)
    count1 = 0
    count2 = 0
    for dirs in data_dir_list:
        img_list=os.listdir(data_path+'\\'+ dirs)[1:]
        count1, count2 =count2, count2 + len(img_list)
        label[count1:count2] = dirs[1:2]
        
    #list of labels    
    poi_list = data_dir_list
      
    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(label, num_classes)
    
    #Shuffle the dataset
    x,y = shuffle(img_data,Y, random_state=2)
    
    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)    
    return X_train, X_test, Y_train, Y_test, poi_list

    

#%%

if __name__ == '__main__':
    
    # path of folder of images
    data_path = 'german_poi_200/' 
    
    img_rows, img_cols = 224, 224 # Resolution of inputs
    num_channel = 3
    num_classes = 10 
    batch_size = 16 
    nb_epoch = 10
    img_mode = 'RGB'
   
    
    X_train, X_test, Y_train, Y_test, poi_list = preprocessing(img_rows, img_cols, data_path)
    
    #Let's train our model
    # Load our model
    model = vgg19_model(img_rows, img_cols, num_channel, num_classes)
    
    # Start Fine-tuning
    hist = model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_test, Y_test),
              )
    
    
     #read data and augment it
#    datagen = ImageDataGenerator (featurewise_center=True,
#                                  featurewise_std_normalization=True,
#                                  rotation_range=20,
#                                  width_shift_range=0.2,
#                                  height_shift_range=0.2,
#                                  horizontal_flip=True,
#                                  rescale=1. / 255,
#                                  shear_range=0.2,
#                                  zoom_range=0.2)
#    
#    #applies ImageDataGenerator to the dataset
#    hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), 
#                        steps_per_epoch = len(X_train)/32, 
#                        epochs = nb_epoch, 
#                        verbose=1,
#                        validation_data=(X_test, Y_test))
    
    # Make predictions
    predictions_valid = model.predict(X_test, batch_size=batch_size, verbose=1)
    
    # Evaluating the model with test loss, test accuracy and top3 accuracy
    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])
    print('Top-3 accuracy:', score[2])
    
    # Computing the error rates 
    error = 1-score[1]
    top3error = 1- score[2]
    
    print('Error: ', error)
    print('Top-3 error: ', top3error)

    
    #Testing the random image from internet
    from skimage import io
    #just insert any direct link to the image
    url = 'http://www.baviere-quebec.org/imperia/md/quebec/tourismus/neuschwanstein_bild.jpeg'
    test_image = io.imread(url)
    
    test_image = imresize(test_image,(img_rows,img_cols))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
       
    if num_channel==1:
    	if K.image_dim_ordering()=='th':
    		test_image= np.expand_dims(test_image, axis=0)
    		test_image= np.expand_dims(test_image, axis=0)
    	else:
    		test_image= np.expand_dims(test_image, axis=3) 
    		test_image= np.expand_dims(test_image, axis=0)
    		
    else:
    	if K.image_dim_ordering()=='th':
    		test_image=np.rollaxis(test_image,2,0)
    		test_image= np.expand_dims(test_image, axis=0)
    	else:
    		test_image= np.expand_dims(test_image, axis=0)
    		
    pred = model.predict(test_image).reshape(10,)	
	
    max_val = np.amax(pred)			
    max_ind = np.argmax(pred)			
    			
    print('Test image belongs to: class 0' + str(max_ind) + ' "' + str(poi_list[max_ind]) + '" with the acccuracy: ' + str(max_val) )		
    
    #Let's see the tested the image
    img = test_image.reshape(img_rows, img_cols, num_channel)
    plt.imshow(img)


