#coding=utf-8
import numpy as np
import struct

def loadImageSet(filename):
    binfile=open(filename,'rb')
    buffers=binfile.read()

    #取前4个整数，返回一个元组
    head=struct.unpack_from('>IIII',buffers,0)

    offset=struct.calcsize('>IIII')
    imgNum=head[1]
    width=head[2]
    height=head[3]

    bits=imgNum*width*height #data 60000*28*28
    bitsString='>'+str(bits)+'B'
    imgs=struct.unpack_from(bitsString,buffers,offset)
    binfile.close()

    imgs=np.reshape(imgs,[imgNum,width*height]) # [600000,784]
    imgs=np.reshape(imgs,(imgNum,1,28,28)).transpose(0,2,3,1).astype("float")

    return imgs,head

def loadLabelSet(filename):
    binfile=open(filename,'rb')
    buffers=binfile.read()

    head=struct.unpack_from('>II',buffers,0)
    labelNum=head[1]
    offset=struct.calcsize('>II')

    numString='>'+str(labelNum)+"B"
    labels=struct.unpack_from(numString,buffers,offset)

    binfile.close()
    labels=np.reshape(labels,[labelNum])

    return labels,head

if __name__=="__main__":
    X_train_path = 'data/train-images-idx3-ubyte'
    y_train_path = 'data/train-labels-idx1-ubyte'
    X_test_path = 'data/t10k-images-idx3-ubyte'
    y_test_path = 'data/t10k-labels-idx1-ubyte'

    X_train,_=loadImageSet(X_train_path)
    X_test,_=loadImageSet(X_test_path)
    y_train,_=loadLabelSet(y_train_path)
    y_test,_=loadLabelSet(y_test_path)

    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape
