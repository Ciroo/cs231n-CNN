#coding:utf-8
import numpy as np

#全连接层
def affine_forward(x,w,b):
    out=None
    num_inputs=x.shape[0]
    out=x.reshape(num_inputs,-1).dot(w)+b
    cache=(x,w,b)
    return out,cache

def affine_backward(dout,cache):
    x,w,b=cache
    num_input=x.shape[0]
    dx,dw,db=None,None,None
    delta=dout
    dw=x.reshape(num_input,-1).transpose().dot(delta)
    db=np.sum(delta,axis=0)
    dx=delta.dot(w.transpose()).reshape(x.shape)
    return dx,dw,db

#relu层
def relu_forward(x):
    out=None
    x_cpy=x.copy()
    x_cpy[x_cpy<=0]=0
    out=x_cpy
    cache=x
    return out,cache

def relu_backward(dout,cache):
    dx,x=None,cache
    indicator=x.copy()
    indicator[x<0]=0
    indicator[x>=0]=1
    dx=dout*indicator
    return dx

#卷积层

def conv_forward(x,w,b,conv_param):
    N,C,H,W=x.shape
    F,C,HH,WW=w.shape

    stride=conv_param['stride']

    pad=conv_param['pad']

    #输出特征图的大小
    H_out=1+(H+2*pad-HH)/stride
    W_out=1+(W+2*pad-WW)/stride

    out=np.zeros((N,F,H_out,W_out))
    x_pad=np.zeros((N,C,H+2*pad,W+2*pad))
    for i in range(N):
        for j in range(C):
            x_pad[i,j]=np.pad(x[i,j],((pad,pad),(pad,pad)),mode='constant')

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    out[n,f,i,j]=np.sum(w[f]*x_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW])+b[f]

    cache=(x,x_pad,w,b,conv_param)
    return out,cache

def conv_backward(dout,cache):
    dx,dw,db=None,None,None
    x,x_pad,w,b,conv_param=cache
    N,C,H,W=x.shape
    F,C,HH,WW=w.shape
    stride=conv_param['stride']
    pad=conv_param['pad']
    _,_,H_out,W_out=dout.shape
    dx_pad=np.zeros(x_pad.shape)
    dw=np.zeros(w.shape)
    db=np.zeros(b.shape)
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    db[f] += dout[n, f, i, j]
                    dw[f] += dout[n, f, i, j] * x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                    dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += dout[n, f, i, j] * w[f]
    dx = dx_pad[:, :, pad:pad + H, pad:pad + W].copy()
    return dx,dw,db

#最大池化层
def max_pool_forward(x,pool_param):
    N,C,H,W=x.shape
    stride=pool_param['stride']
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    H_out=1+(H-pool_height)/stride
    W_out=1+(W-pool_width)/stride

    out=np.zeros((N,C,H_out,W_out))
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    out[n,c,i,j]=np.max(x[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width])

    cache=(x,pool_param)
    return out,cache

def max_pool_backward(dout,cache):
    x,pool_param=cache
    dx=np.zeros(x.shape)
    N,C,H,W=x.shape
    stride=pool_param['stride']
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    H_out=1+(H-pool_height)/stride
    W_out=1+(W-pool_width)/stride
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    index = np.argmax(x[n, c, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width])
                    index = np.unravel_index(index, (pool_height, pool_width))

                    dx[n, c, index[0] + i * stride, index[1] + j * stride] += dout[n, c, i, j]


    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  mode = bn_param['mode'] #train or test
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)
  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))#test时的mean
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))#test时的var
  out, cache = None, None
  if mode == 'train':
      #求mini-batch的均值、方差、归一数据
      means = np.mean(x,axis=0)
      std = np.std(x,axis=0)
      x_ =(x - np.tile(means,(N,1)))/np.sqrt(np.tile(std,(N,1))**2+eps)
      #平移、放缩
      bn_x = x_*gamma + beta
      running_mean = momentum*running_mean + (1-momentum)*means
      running_var = momentum*running_var + (1-momentum)*(std**2)
      out = bn_x
      cache = (means,std**2,x,x_,bn_x,gamma,beta,eps)
  elif mode == 'test':#测试时使用running_mean和running_var作为均值和方差
      x_ = (x-np.tile(running_mean,(N,1)))/np.sqrt(np.tile(running_var,(N,1))+eps)
      out = x_*gamma + beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache

#BN的反向过程，按论文的公式计算
def batchnorm_backward(dout, cache):
  N,D = dout.shape
  means,var,x,x_,bn_x,gamma,beta,eps = cache
  dx_ = dout*gamma
  #对方差的导数
  dvar = np.sum((dx_*(x-means)*(-0.5)*np.power((var+eps),-3.0/2)),axis=0)
  #对均值的导数
  dmeans = np.sum((dx_*(-1./np.sqrt(var+eps))),axis=0) + dvar*np.sum(-2*(x-means),axis=0)/N
  #误差
  dx = dx_*(1./np.sqrt(var+eps)) + dvar*(2*(x-means)/N) + dmeans/N
  #放缩系数的导数
  dgamma = np.sum(dout*x_,axis=0)
  #平移系数的导数
  dbeta = np.sum(dout,axis=0)
  return dx, dgamma, dbeta
#dropout的前向过程
def dropout_forward(x, dropout_param):
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])
  mask = None
  out = None
  #仅在训练时使用dropout
  if mode == 'train':
    retain_p = 1-p
    mask = np.random.binomial(1,retain_p,x.shape)
    out = x*mask/retain_p
  elif mode == 'test':
    out = x
  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)
  return out, cache

#dropout的反向过程
def dropout_backward(dout, cache):
  dropout_param, mask = cache
  mode = dropout_param['mode']
  drop_p = dropout_param['p']
  retain_p = 1-drop_p
  dx = None
  if mode == 'train':
      #只对那些保留的神经元传递误差
      dx = dout*mask
  elif mode == 'test':
      dx = dout
  return dx


# 卷积层的BN前馈过程
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    N, C, H, W = x.shape
    x_cpy = x.copy()

    out, cache = batchnorm_forward(np.transpose(x, (0, 2, 3, 1)).reshape(-1, C), gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


# 卷积层的BN反向过程
def spatial_batchnorm_backward(dout, cache):
    N, C, H, W = dout.shape
    dout_cpy = dout.copy()
    dout = np.transpose(dout, (0, 2, 3, 1)).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta

#softmax
def softmax_loss(x, y):
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx












