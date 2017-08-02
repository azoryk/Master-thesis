Requirements: 
Python 3.6

For using GPU (training with GPU is up to 75 times faster): 
CUDA Toolkit 8.0
CUDNN V5.1 

Required Python modules: 
keras
tensorflow
tensorflow-gpu
flickrapi
numpy
scipy
sklearn
matplotlib

german_poi_dataset.py script is for downloading images from Flickr. The searching is by the tags provided in the german_poi.txt list. Put a new tag in the new line. The list of urls will be saved in txt-file and then downloaded to specific folder. 

Folder "german_poi_200" contains small dataset of 10 German landmarks. 

Folder 'model_weights' contains pre-trained weights for each of the models. 

All the programs are ready to use with the small dataset "german_poi_200". For large dataset "German POI 1K" please download it by the following url https://drive.google.com/open?id=0B2AYM9nQJq_HcUFTM1E3cWNkZHc to this folder and change the path in script. 

To use resnet_model_event.py, the extended dataset "German POI 1K+" is needed. Please download it by the following url https://drive.google.com/open?id=0B2AYM9nQJq_HU2JIV1FHOTdFWTg to this folder and put the correct value for data_path attribute. 





