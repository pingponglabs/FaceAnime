# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:17:35 2017

@author: zhs
"""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import os, cv2
import numpy as np
import scipy.misc
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    labels = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
#                    item = (path, class_to_idx[target])
                    labels.append(class_to_idx[target])

    return images,labels

def get_images_labels(root):
    
    classes, class_to_idx = find_classes(root)
    
    images,labels = make_dataset(root, class_to_idx)
    
    return images,labels,classes
    
        
        
        
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
images,labels,classes = get_images_labels('./ijbc_train_120x120_max80_refine/')

import pdb
f = open("./xxxxx.txt","w+")
for index in range(len(images)):
    print(index)
    iimage = cv2.imread(images[index])
    image = scipy.misc.imread(images[index]).astype(np.float)
    if len(np.shape(image))==3:
        #pdb.set_trace()
        aa = images[index].split('_refine/') 
        f.write(aa[1] +' '+str(labels[index])+'\r\n')
    
f.close()
    
