#coding=utf-8
from read_data import loadImageSet,loadLabelSet
import numpy as np
def get_MNIST_data(num_validation=1000):
    X_train_path = 'data/train-images-idx3-ubyte'
    y_train_path = 'data/train-labels-idx1-ubyte'
    X_test_path = 'data/t10k-images-idx3-ubyte'
    y_test_path = 'data/t10k-labels-idx1-ubyte'

    X_train, _ = loadImageSet(X_train_path)
    X_test, _ = loadImageSet(X_test_path)
    y_train, _ = loadLabelSet(y_train_path)
    y_test, _ = loadLabelSet(y_test_path)


    mask=range(num_validation)
    X_val=X_train[mask]
    y_val=y_train[mask]
    X_train=X_train[len(mask):]
    y_train=y_train[len(mask):]

    #减去均值
    mean_image=np.mean(X_train,axis=0)
    X_train-=mean_image
    X_val-=mean_image
    X_test-=mean_image

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return {
        'X_train':X_train,'y_train':y_train,
        'X_val':X_val,'y_val':y_val,
        'X_test':X_test,'y_test':y_test,
    }






if __name__=="__main__":
    d=get_MNIST_data()
    print d['X_test'].shape
