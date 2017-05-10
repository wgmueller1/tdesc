
#!/usr/bin/env python

"""
    workers.py
"""

import os
import sys
import urllib
import cStringIO
import numpy as np

# VGG16 featurization
from keras.applications import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# Face featurization
import dlib
from skimage import io

class VGG16Worker(object):
    
    def __init__(self, crow, target_dim=224):
        if crow:
            self.model = VGG16(weights='imagenet', include_top=False)
        else:
            whole_model = VGG16(weights='imagenet', include_top=True)
            self.model = Model(inputs=whole_model.input, outputs=whole_model.get_layer('fc2').output)
        
        self.target_dim = target_dim
        self.crow = crow
        
        self._warmup()
    
    def _warmup(self):
        _ = self.model.predict(np.zeros((1, self.target_dim, self.target_dim, 3)))
    
    def imread(self, path):
        if 'http' == path[:4]:
            img = cStringIO.StringIO(urllib.urlopen(path).read())
            img = image.load_img(img, target_size=(self.target_dim, self.target_dim))
        else:
            img = image.load_img(path, target_size=(self.target_dim, self.target_dim))
        
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
    
    def featurize(self, path, img):
        feat = self.model.predict(img)
        if self.crow:
            feat = feat.sum(axis=(0, 1))
        else:
            feat = feat.squeeze()
        
        print '\t'.join((path, '\t'.join(map(str, feat))))
    
    def close(self):
        pass

class DlibFaceWorker(object):
    
    def __init__(self, shape_path, face_path, out_path):
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(shape_path)
        self.facerec = dlib.face_recognition_model_v1(face_path)
        self.db = h5py.File(out_path)
    
    def imread(self, path):
        img = io.imread(path)
        dets = detector(img, 1)
        return img, dets
    
    def featurize(self, path, obj):
        img, dets = obj
        for k,d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape, 10)
            self.db['%s/%d/feat' % (os.path.basename(line), k)] = np.array(face_descriptor)
            self.db['%s/%d/img' % (os.path.basename(line), k)] = img[d.top():d.bottom(),d.left():d.right()]
    
    def close(self):
        self.db.close()

