# -*- coding: utf-8 -*-
'''
This script trains the Xception model on the small dataset "German POI 200"

The source of model was taken by the followning link: 
https://github.com/fchollet/deep-learning-models/blob/master/xception.py

Pretrained weights were downloaded by this link:
'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'

# Reference:
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

@author: Andriy Zoryk
'''

#%%
from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.optimizers import SGD
from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.applications.imagenet_utils import _obtain_input_shape

from keras.preprocessing.image import ImageDataGenerator

#from sklearn.metrics import log_loss

TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

#%%
def Xception_model(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the Xception architecture.

    Optionally loads weights pre-trained
    on ImageNet. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    You should set `image_data_format="channels_last"` in your Keras config
    located at ~/.keras/keras.json.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if K.backend() != 'tensorflow':
        raise RuntimeError('The Xception model is only available with '
                           'the TensorFlow backend.')
    if K.image_data_format() != 'channels_last':
        warnings.warn('The Xception model is only available for the '
                      'input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height). '
                      'You should set `image_data_format="channels_last"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=71,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='xception')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)

    
    if old_data_format:
        K.set_image_data_format(old_data_format)
#    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
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

     # path of folder of images, better to provide full path in binary form
    data_path = 'german_poi_200/'
    
    # Example to fine-tune on 3000 samples from Cifar10
    img_rows, img_cols = 299, 299 # Resolution of inputs
    num_channel = 3
    num_classes = 10 
    batch_size = 16 
    nb_epoch = 10
    img_mode = 'RGB'
    
    #Let's get our train and test sets of images and labels, and list of labels
    X_train, X_test, Y_train, Y_test, poi_list = preprocessing(img_rows, img_cols, data_path)

    #pre-trained CNN Model using imagenet dataset for pre-trained weights
    base_model = Xception_model( input_shape=( img_rows, img_cols, num_channel), weights='imagenet', include_top=False)
    
    #Top model block
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)
        
    #add the top layer to the base model
    model = Model(base_model.input, predictions)
    print(model.summary())
    
    
    #let's train only top layers and freeze all layers that were pre-trained
    for layer in base_model.layers:
        layer.trainable = False
    
    #read data and augment it
    datagen = ImageDataGenerator (featurewise_center=True,
                                  featurewise_std_normalization=True,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  horizontal_flip=True,
                                  rescale=1. / 255,
                                  shear_range=0.2,
                                  zoom_range=0.2)
    
    # Defines the metrics for calculating top3 accuracy
    from keras.metrics import top_k_categorical_accuracy
    def top_3_categorical_accuracy(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3) 

    # Compiles the model with stochastic gradient descent
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', top_3_categorical_accuracy ])
    
    hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), 
                        steps_per_epoch = len(X_train)/32, 
                        epochs = nb_epoch, 
                        verbose=1,
                        validation_data=(X_test, Y_test))
    
    # Start Fine-tuning
#    hist = model.fit(X_train, Y_train,
#              batch_size=batch_size,
#              epochs=nb_epoch,
#              shuffle=True,
#              verbose=1,
#              validation_data=(X_test, Y_test),
#              )

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
    
    #Let's see the tested image
    img = test_image.reshape(img_rows, img_cols, num_channel)
    plt.imshow(img)