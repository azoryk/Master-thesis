# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:26:40 2017

@author: Administrator
"""

#%%
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss

#from load_cifar10 import load_cifar10_data


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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
#from keras.applications.vgg16 import VGG16

import warnings

from keras.models import Model

from keras.layers import Input
from keras.layers import Conv2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

from sklearn.metrics import log_loss

#%%

def vgg19_model(img_rows, img_cols, channel=1, num_classes=None):
    """
    VGG 19 Model for Keras
    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilZ2RVeVhKY0FyRmc/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    """
  
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(channel, img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
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
    model.load_weights(r'C:\Users\Administrator\Desktop\az\vgg19_weights_tf_dim_ordering_tf_kernels.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


#%%


#%%

if __name__ == '__main__':

    
    # path of folder of images
    data_path = r'C:\Users\Administrator\Desktop\az\poi_dataset_200\train' 
    
    # Define data path
    
    data_dir_list = os.listdir(data_path) #MacOS creates automatically '.DS_Store' file in each folder
    if data_dir_list[0] == '.DS_Store':
        data_dir_list = os.listdir(data_path)[1:]
    print (data_dir_list)
    
    img_rows=224 #resolution of inputs
    img_cols=224
    num_channel=3
    img_mode = 'RGB'
    batch_size = 16 
    nb_epoch=20
    
    
    # Define the number of classes
    num_classes = 10   
    
    
    #lets make the list of all the images
    img_data_list=[]   
    #let's count the total number of images
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
                
                #input_img_grey=input_img.convert('L')
                input_img_resize = imresize(input_img, (img_rows, img_cols))
                img_data_list.append(input_img_resize)
    
    print(num_samples)
    
    print(len(img_data_list))
    
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

  
    
    label = np.ones((num_samples,), dtype=int)
    count1 = 0
    count2 = 0
    for dirs in data_dir_list:
        img_list=os.listdir(data_path+'\\'+ dirs)[1:]
        count1, count2 =count2, count2 + len(img_list)
        label[count1:count2] = dirs[1:2]
        
    #list of labels    
    poi_list = ['neuschwanstein','cologne cathedral','brandenburger tor', 'heidelberg castle' , 
                'marienplatz', 'frauenkirche dresden', 'berlin wall', 'reichstag', 
                'nymphenburg', 'speicherstadt']
    
    
    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(label, num_classes)
    
    #Shuffle the dataset
    x,y = shuffle(img_data,Y, random_state=2)
    
    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    
    
    
#%%    
    
#Let's train our model
    # Load our model
    model = vgg19_model(img_rows, img_cols, num_channel, num_classes)

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_test, Y_test),
              )

    # Make predictions
    predictions_valid = model.predict(X_test, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_test, predictions_valid)
    
    
    #%%

# Evaluating the model

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[12:13]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(Y_test[12:13])

#
#img = test_image.reshape(img_rows, img_cols)
#plt.imshow(img)
#plt.imshow(img, cmap='gray')
