# -*- coding: utf-8 -*-
"""
This script trains the ResNet-50 model on the big extended dataset "German POI 1K+"

This dataset should be downloaded from 
https://drive.google.com/open?id=0B2AYM9nQJq_HU2JIV1FHOTdFWTg

Please provide the path to your dataset folder

The source of model was taken by the followning link: 
https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

@author: Andriy Zoryk
"""

# -*- coding: utf-8 -*-
#%%
import os
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, merge, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

#%%

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def resnet50_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    Resnet 50 Model for Keras
    Model Schema is based on 
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
    ImageNet Pretrained Weights 
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5
    #Arguments
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    # Returns
       A Keras model instance.
    """

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type))
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Fully Connected Softmax Layer
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # Create model
    model = Model(img_input, x_fc)

#    # Load ImageNet pre-trained data 
    weights_path = 'model_weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)

    # Create another model with our customized softmax
    model = Model(img_input, x_newfc)

    #metrics for top3 accuracy
    from keras.metrics import top_k_categorical_accuracy
    def top_3_categorical_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3) 

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', top_3_categorical_accuracy ])
  
    #overview of the parameters of model
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
        label[count1:count2] = dirs[1:3]
        
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

    
    img_rows, img_cols = 224, 224 # Resolution of inputs
    num_channel = 3
    num_classes = 12 
    batch_size = 16 
    nb_epoch = 20
    img_mode = 'RGB'


################################################################
#Change the path to the dataset 
#The extended dataset "German POI 1K+" can be downloaded by the following url 
# https://drive.google.com/open?id=0B2AYM9nQJq_HU2JIV1FHOTdFWTg
# Be aware that for this program 12 classes are considered
################################################################
    # provide path to folder with dataset
    data_path = 'German POI 1K+/'  
    
    #import our train and test sets of images and labels, and list of labels
    X_train, X_test, Y_train, Y_test, poi_list = preprocessing(img_rows, img_cols, data_path)
    model = resnet50_model(img_rows, img_cols, num_channel, num_classes)

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
    url = 'http://www.opelrent.de/img/artikel/facebook/christkindlesmarkt.jpg'
    test_image = io.imread(url)
    
    
    #test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = imresize(test_image,(img_rows,img_cols))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    #print (test_image.shape)
       
    if num_channel==1:
    	if K.image_dim_ordering()=='th':
    		test_image= np.expand_dims(test_image, axis=0)
    		test_image= np.expand_dims(test_image, axis=0)
    		#print (test_image.shape)
    	else:
    		test_image= np.expand_dims(test_image, axis=3) 
    		test_image= np.expand_dims(test_image, axis=0)
    		#print (test_image.shape)
    		
    else:
    	if K.image_dim_ordering()=='th':
    		test_image=np.rollaxis(test_image,2,0)
    		test_image= np.expand_dims(test_image, axis=0)
    		#print (test_image.shape)
    	else:
    		test_image= np.expand_dims(test_image, axis=0)
    		#print (test_image.shape)
    		
    pred = model.predict(test_image).reshape(12,)	
    #print(pred)			
    			
    max_val = np.amax(pred)			
    max_ind = np.argmax(pred)			
    			
    print('Test image belongs to: class 0' + str(max_ind) + ' "' + str(poi_list[max_ind]) + '" with the acccuracy: ' + str(max_val) )		
    
    img = test_image.reshape(img_rows, img_cols, num_channel)
    plt.imshow(img)



