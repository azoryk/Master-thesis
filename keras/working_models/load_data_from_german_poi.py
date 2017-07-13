#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:12:23 2017

@author: azoryk
"""

import os
from scipy.misc import imread, imresize
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_data_from_german_poi(data_path, img_rows, img_cols, img_mode='RGB', num_channel=3):
    
    """This loads the data from the folder with dataset
    
    #Arguments
        data_path: 
        img_rows, img_cols:  
        img_mode: 'RGB' or 'L' for greyscale
        num_channel: 3 for rgb, 1 for greyscale
        
        
    #Returns
        4 output tensors:
        X_train - training set of images
        Y_train - training set of labels
        X_test - test set of images
        Y_test - test set of labels
    
    """
    
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
        img_list=os.listdir(data_path+'/'+ dataset)
        if img_list[0] == '.DS_Store':
            img_list = os.listdir(data_path+'/'+ dataset)[1:] 
        num_samples +=len(img_list)
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        
        for img in img_list:
                input_img= imread(data_path + '/'+ dataset + '/'+ img, mode = img_mode)
                
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
    
    # Define the number of classes
    num_classes = 10     
    
    label = np.ones((num_samples,), dtype=int)
    count1 = 0
    count2 = 0
    for dirs in data_dir_list:
        if img_list[0] == '.DS_Store':
            img_list = os.listdir(data_path+'/'+ dirs)[1:] 
        count1, count2 =count2, count2 + len(img_list)
        label[count1:count2] = dirs[1:2]
        
    #list of labels    
#    poi_list = ['neuschwanstein','cologne cathedral','brandenburger tor', 'heidelberg castle' , 
#                'marienplatz', 'frauenkirche dresden', 'berlin wall', 'reichstag', 
#                'nymphenburg', 'speicherstadt']
    
    
    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(label, num_classes)
    
    #Shuffle the dataset
    x,y = shuffle(img_data,Y, random_state=2)
    
    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)    
    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    #X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    
    #shuffle images
    data, Label = shuffle(img_data, label, random_state=2)
    train_data = [data, Label]
    #print(train_data[0].shape)
    #print(train_data[1].shape)
    
    return X_train, X_test, Y_train, Y_test

def label_list(data_path):
    
    data_dir_list = os.listdir(data_path) #MacOS creates automatically '.DS_Store' file in each folder
    if data_dir_list[0] == '.DS_Store':
        data_dir_list = os.listdir(data_path)[1:]
    
    label_list = []
    for elem in data_dir_list: 
        elem = elem[3:]
        label_list.append(elem)
     
    return label_list


X_train, X_test, Y_train, Y_test = load_data_from_german_poi('/Users/azoryk/PycharmProjects/small_dataset/input', 224, 224)
print(Y_train.shape)
