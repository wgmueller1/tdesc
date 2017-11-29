# -*- coding: utf-8 -*-


import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import glob
import pandas as pd

tf.__version__

PATH = os.getcwd()

LOG_DIR = PATH+ '/embedding-logs'
#metadata = os.path.join(LOG_DIR, 'metadata2.tsv')

#%%
#feature_vectors = np.loadtxt('vgg16-crow.descriptors',usecols=(1,2,3,4),delimiter='\t')


def featurize(image):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
 
        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
# divide the image into four rectangles/segments (top-left,
        # top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
            (0, cX, cY, h)]
 
        # construct an elliptical mask representing the center of the
        # image
        (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
 
        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
 
            # extract a color histogram from the image, then update the
            # feature vector
            hist = histogram(image, cornerMask)
            features.extend(hist)
 
        # extract a color histogram from the elliptical region and
        # update the feature vector
        hist = histogram(image, ellipMask)
        features.extend(hist)
 
        # return the feature vector
        return features


def histogram(image, mask):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel; then
        # normalize the histogram
        hist = cv2.calcHist([image], [0, 1, 2], mask, (8,12,3),
            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist,hist).flatten()
 
        # return the histogram
        return hist

import cv2
import numpy as np
img_data=[]
#img_list=glob.glob('/Users/wgmueller/SocialSim/reddit/images/*.*')
with open('files2','r') as f:
    img_list=f.readlines()

print ('Loaded the images of dataset')
names=[]
features=[]
for img in img_list:
    try:
        names.append('reddit/images/'+img.split('/')[-1].strip())
        print('reddit/images/'+img.split('/')[-1])
        input_img=cv2.imread(img.strip())
        input_img_resize=cv2.resize(input_img,(80,80)) # you can choose what size to resize your data
        features.append(featurize(input_img))
        img_data.append(input_img_resize)
    except Exception as e:
        print(e)
        
img_data = np.array(img_data)

feature_vectors = pd.read_csv('vgg16.descriptors',sep='\t',header=None)
feature_vectors = feature_vectors[feature_vectors[0].isin(names)]
feature_vectors.set_index(0,inplace=True)
feature_vectors.loc[names,:]
feature_vectors = feature_vectors.values
features = tf.Variable(features, name='features')



print ("feature_vectors_shape:",feature_vectors.shape)
print ("num of images:",feature_vectors.shape[0])
print ("size of individual feature vector:",feature_vectors.shape[1])


num_of_samples=feature_vectors.shape[0]
num_of_samples_each_class = 100




#%%



#y = np.ones((num_of_samples,),dtype='int64')

#y[0:100]=0
#y[100:200]=1
#y[200:300]=2
#y[300:]=3

#names = ['cats','dogs','horses','humans']

#with open(metadata, 'w') as metadata_file:
#    for row in range(210):
#        c = y[row]
#        metadata_file.write('{}\n'.format(c))
#metadata_file = open(os.path.join(LOG_DIR, 'metadata_4_classes.tsv'), 'w')
#metadata_file.write('Class\tName\n')
#k=100 # num of samples in each class
#j=0
#for i in range(210):
#    metadata_file.write('%06d\t%s\n' % (i, names[y[i]]))
#for i in range(num_of_samples):
#    c = names[y[i]]
#    if i%k==0:
 #       j=j+1
#    metadata_file.write('{}\t{}\n'.format(j,c))
    #metadata_file.write('%06d\t%s\n' % (j, c))
#metadata_file.close()
       
    
# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data
#%%
sprite = images_to_sprite(img_data)
cv2.imwrite(os.path.join(LOG_DIR, 'sprite_4_classes.png'), sprite)
#scipy.misc.imsave(os.path.join(LOG_DIR, 'sprite.png'), sprite)

#%%
with tf.Session() as sess:
    saver = tf.train.Saver([features])

    sess.run(features.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images_4_classes.ckpt'))
    
    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    # Link this tensor to its metadata file (e.g. labels).
    #embedding.metadata_path = os.path.join(LOG_DIR, 'metadata_4_classes.tsv')
    # Comment out if you don't want sprites
    embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_4_classes.png')
    embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
