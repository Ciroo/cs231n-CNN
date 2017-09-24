#coding:utf-8
import numpy as np
import sys
from layers.layers import *
from layers.layer_utils import *

class MultiLayerConvNet(object):

    def __init__(self,input_dim,num_filters,num_hids,filter_size,
                 num_classes,weight_scale,reg,conv_params,pool_params,dtype,use_batchnorm):
        #conv_layers
        num_cp=len(num_filters)
        #hidden_layers
        num_hid=len(num_hids)

        self.params={}
        self.reg=reg
        self.dtype=dtype
        #conv
        self.num_cp=num_cp
        #hidden
        self.num_hid=num_hid

        #总的层数
        self.hidlayers=num_cp+num_hid+1

        #input
        C,H,W=input_dim

        self.conv_params=conv_params
        self.pool_params=pool_params

        self.bn_params = []
        self.use_batchnorm = use_batchnorm

        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.hidlayers - 1)]

        #初始化卷积层
        for i in range(num_cp):
            if(i==0):
                self.params['W'+str(i+1)]=weight_scale*np.random.randn(num_filters[i],
                                                                       C,filter_size[i],filter_size[i])
            else:
                self.params['W'+str(i+1)]=weight_scale*np.random.randn(num_filters[i],
                                                                       num_filters[i-1],filter_size[i],filter_size[i])
            self.params['b'+str(i+1)]=np.zeros((num_filters[i]))

            if self.use_batchnorm:
                # 初始化gamma
                self.params['gamma' + str(i + 1)] = np.ones(num_filters[i])
                # 初始化deta
                self.params['beta' + str(i + 1)] = np.zeros(num_filters[i])

        #初始化全连接层
        for i in range(num_hid):
            if(i==0):
                #self.params['W'+str(num_cp+i+1)]=weight_scale*np.random.randn(num_filters[-1]*H*W/np.power(2,len(pool_params)*2),num_hids[i])
                self.params['W' + str(num_cp + i + 1)] = weight_scale * np.random.randn(512, num_hids[i])

            else:
                self.params['W' + str(num_cp + i + 1)] = weight_scale * np.random.randn(num_hids[i-1], num_hids[i])
            self.params['b' + str(num_cp + i + 1)] = np.zeros((num_hids[i]))
            if self.use_batchnorm:
                self.params['gamma' + str(num_cp + i + 1)] = np.ones(num_hids[i])
                self.params['beta' + str(num_cp + i + 1)] = np.zeros(num_hids[i])

        # 初始化最后一层
        self.params['W' + str(self.hidlayers)] = weight_scale * np.random.randn(num_hids[-1], num_classes)
        self.params['b' + str(self.hidlayers)] = np.zeros((num_classes))
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self,X,y=None):
        X=X.astype(self.dtype)

        mode='test' if y is None else 'train'

        sum_w=0.0
        grads={}

        if self.use_batchnorm:
            for i in range(self.hidlayers - 1):
                self.bn_params[i]['mode'] = mode

        cp_data=X
        cp_hid_layer_out=[]
        cp_hid_layer_cache=[]

        for i in range(self.num_cp):
            out,cache=conv_bn_relu_pool_forward(cp_data,self.params['W'+str(i+1)],
                                        self.params['b'+str(i+1)],self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)],
                                        self.conv_params[i],
                                        self.pool_params[i],
                                                self.bn_params[i])
            sum_w += 0.5 * self.reg * (np.sum(cache[0][1] ** 2))
            cp_hid_layer_out.append(out)
            cp_hid_layer_cache.append(cache)
            cp_data=out

        fc_hid_layer_out=[]
        fc_hid_layer_cache=[]
        fc_data=cp_hid_layer_out[-1]

        for i in range(self.num_hid):
            out,cache=affine_bn_forward(fc_data,
                                            self.params['W'+str(self.num_cp+i+1)],
                                            self.params['b'+str(self.num_cp+i+1)],
                                            self.params['gamma' + str(self.num_cp + i + 1)],
                                            self.params['beta' + str(self.num_cp + i + 1)],
                                            self.bn_params[self.num_cp + i])

            sum_w += 0.5 * self.reg * np.sum(cache[0][1] ** 2)
            fc_hid_layer_out.append(out)
            fc_hid_layer_cache.append(cache)
            fc_data=out
        #最后一层
        final_out=affine_forward(fc_hid_layer_out[-1],
                                 self.params['W'+str(self.hidlayers)],
                                 self.params['b'+str(self.hidlayers)])
        scores,final_cache=final_out
        sum_w += 0.5 * self.reg * np.sum(final_cache[1] ** 2)
        if y is None:
            return scores

        #计算损失和误差
        loss,dx=softmax_loss(scores,y)
        loss += sum_w

        #最后一层损失
        dx,dw,db=affine_backward(dx,final_cache)

        grads['W' + str(self.hidlayers)] = dw + self.reg * self.params['W' + str(self.hidlayers)]
        grads['b' + str(self.hidlayers)] = db

        fc_delta=dx
        for i in reversed(range(self.num_hid)):
            dx, dw, db, dgamma, dbeta = affine_bn_backward(fc_delta, fc_hid_layer_cache[i])
            grads['gamma' + str(self.num_cp + i + 1)] = dgamma
            grads['beta' + str(self.num_cp + i + 1)] = dbeta


            grads['W' + str(self.num_cp + i + 1)] = dw + self.reg * self.params['W' + str(self.num_cp + i + 1)]
            grads['b' + str(self.num_cp + i + 1)] = db
            fc_delta = dx

        cp_delta=fc_delta
        #卷积层的反向过程
        for i in reversed(range(self.num_cp)):
            dx, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(cp_delta, cp_hid_layer_cache[i ])
            grads['gamma' + str(i+1)] = dgamma
            grads['beta' + str(i+1)] = dbeta
            grads['W' + str(i+1)] = dw + self.reg * self.params['W' + str(i+1)]
            grads['b' + str(i+1)] = db
            cp_delta = dx
        return loss,grads







