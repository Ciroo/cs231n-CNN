#coding:utf
from layers import *
from fast_layers import *

#fc+relu

def affine_relu_forward(x,w,b):
    a,fc_cache=affine_forward(x,w,b)
    out,relu_cache=relu_forward(a)
    cache=(fc_cache,relu_cache)
    return out,cache

def affine_relu_backward(dout,cache):
    fc_cache,relu_cache=cache
    da=relu_backward(dout,relu_cache)
    dx,dw,db=affine_backward(da,fc_cache)
    return dx,dw,db

#conv+pool

def conv_relu_pool_forward(x,w,b,conv_param,pool_param):
    a,conv_cache=conv_forward_fast(x,w,b,conv_param)
    s,relu_cache=relu_forward(a)
    out,pool_cache=max_pool_forward_fast(s,pool_param)
    cache=(conv_cache,relu_cache,pool_cache)
    return out,cache

def conv_relu_pool_backward(dout,cache):
    conv_cache,relu_cache,pool_cache=cache
    ds=max_pool_backward_fast(dout,pool_cache)
    da=relu_backward(ds,relu_cache)
    dx,dw,db=conv_backward_fast(ds,conv_cache)
    return dx,dw,db

#conv+bn+relu+pool
def conv_bn_relu_pool_forward(x, w, b,gamma,beta ,conv_param, pool_param,bn_param):
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  sbn_out, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  relu_out, relu_cache = relu_forward(sbn_out)
  pool_out, pool_cache = max_pool_forward_fast(relu_out,pool_param)
  cache = (conv_cache, relu_cache, pool_cache,sbn_cache)
  return pool_out,cache

def conv_bn_relu_pool_backward(dout, cache):
  conv_cache, relu_cache, pool_cache,sbn_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dsbn,dgamma,dbeta = spatial_batchnorm_backward(da,sbn_cache)
  dx, dw, db = conv_backward_fast(dsbn,conv_cache)
  return dx, dw, db,dgamma,dbeta


def affine_bn_forward(x, w, b,gamma,beta,bn_param):
    a, fc_cache = affine_forward(x, w, b)
    bn_out,bn_cache = batchnorm_forward(a,gamma,beta,bn_param)
    cache = (fc_cache,bn_cache)
    return bn_out,cache


def affine_bn_backward(dout, cache):
    fc_cache,bn_cache = cache
    dx_bn, dgamma, dbeta = batchnorm_backward(dout,bn_cache)
    dx,dw,db = affine_backward(dx_bn,fc_cache)
    return dx,dw,db,dgamma,dbeta