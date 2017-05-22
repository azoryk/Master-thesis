print("Importing libraries...")
from scipy.misc import imread, imresize
import os



# color_type = 1 - gray
# color_type = 3 - RGB
def get_im_skipy(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = imread(path, True)
    elif color_type == 3:
        img = imread(path)
    # Reduce size
    resized = imresize(img, (img_cols, img_rows))
    return resized


#label all images
# returns dict where key - name of poi, value - image

def label_poi_images():
    dr = dict()
    import os
    cwd = os.getcwd()
    path = os.path.join(cwd, 'small_dataset', 'input', 'train')
    labels = os.listdir(path)
    print(labels)
    print('Labeling POI data')
    for poi in labels:
        print('Load folder c{}'.format(poi))
        imgs = os.listdir(os.path.join(path, poi))
        print(imgs)
        for im in imgs:
            im_path = path + '\\' + poi + '\\' + im
            dr[im_path.split("\\")[-1]] = poi
    print(len(dr))

    return dr








