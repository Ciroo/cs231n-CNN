#coding:utf-8
import numpy as np
import os
import cPickle
from PIL import Image
from read_data import *
def LoadModel(filename):
    f = open(filename,'rb')
    model = cPickle.load(f)
    f.close()
    return model
if __name__ == "__main__":
    train_path = 'data/train-images-idx3-ubyte'
    test_path = 'sample_image'
    X_train,_= loadImageSet(train_path)
    mean = np.mean(X_train,axis=0)
    print mean.shape
    model = LoadModel('paramdata/model')
    for image in os.listdir(test_path):
        img = np.array(Image.open(os.path.join(test_path,image)).convert('L')).reshape(1,1,28,28).transpose(0,2,3,1).astype("float")
        img = img - mean
        img = img.transpose(0, 3, 1, 2).copy()
        scores = model.loss(img)
        y_pred = np.argmax(scores, axis=1)[0]
        print 'image: '+image+', pred label: '+str(y_pred)