#coding:utf-8
import numpy as np
import cPickle
import os
import optim


class Solver(object):
  '''
    @model 我们训练好的model
    @data 样本数据
    @kwargs 保存训练model时的一些参数：如update_rule，batch_size等
  '''
  def __init__(self, model, data, **kwargs):
    self.model = model
    self.X_train = data['X_train']
    self.y_train = data['y_train']
    self.X_val = data['X_val']
    self.y_val = data['y_val']
    
    #读取训练过程的参数
    #更新规则
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    #学习率的decay
    self.lr_decay = kwargs.pop('lr_decay', 1.0)
    #batch_size大小
    self.batch_size = kwargs.pop('batch_size', 100)
    #训练多少轮
    self.num_epochs = kwargs.pop('num_epochs', 10)

    self.print_every = kwargs.pop('print_every', 10)
    self.verbose = kwargs.pop('verbose', True)

    # Throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
      extra = ', '.join('"%s"' % k for k in kwargs.keys())
      raise ValueError('Unrecognized arguments %s' % extra)
    if not hasattr(optim, self.update_rule):
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(optim, self.update_rule)

    self._reset()

  #重新reset一些变量
  def _reset(self):
    #轮数
    self.epoch = 0
    #训练过程中最好的准确率
    self.best_val_acc = 0
    #最优参数
    self.best_params = {}
    #历史loss
    self.loss_history = []
    #历史训练准确率
    self.train_acc_history = []
    #历史验证准确率
    self.val_acc_history = []
    self.optim_configs = {}
    for p in self.model.params:
      d = {k: v for k, v in self.optim_config.iteritems()}
      self.optim_configs[p] = d
  #训练一次
  def _step(self):
    num_train = self.X_train.shape[0]
    batch_mask = np.random.choice(num_train, self.batch_size)
    #筛选一个mini-batch
    X_batch = self.X_train[batch_mask]
    y_batch = self.y_train[batch_mask]
    #计算loss和梯度
    loss, grads = self.model.loss(X_batch, y_batch)
    self.loss_history.append(loss)
    #根据优化策略更新参数
    for p, w in self.model.params.iteritems():
      dw = grads[p]
      config = self.optim_configs[p]
      next_w, next_config = self.update_rule(w, dw, config)
      self.model.params[p] = next_w
      self.optim_configs[p] = next_config

  #计算准确率
  def check_accuracy(self, X, y, num_samples=None, batch_size=100):

    N = X.shape[0]
    if num_samples is not None and N > num_samples:
      mask = np.random.choice(N, num_samples)
      N = num_samples
      X = X[mask]
      y = y[mask]

    #batch的个数
    num_batches = N / batch_size
    if N % batch_size != 0:
      num_batches += 1
    y_pred = []
    for i in xrange(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      scores = self.model.loss(X[start:end])
      y_pred.append(np.argmax(scores, axis=1))
    y_pred = np.hstack(y_pred)
    acc = np.mean(y_pred == y)

    return acc


  def train(self):
    
    print 'train'
    num_train = self.X_train.shape[0]
    #每轮迭代的次数
    iterations_per_epoch = max(num_train / self.batch_size, 1)
    #总的迭代次数
    num_iterations = self.num_epochs * iterations_per_epoch

    for t in xrange(num_iterations):
      self._step()

      if self.verbose and t % self.print_every == 0:
        print '(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1])

      epoch_end = (t + 1) % iterations_per_epoch == 0
      #一轮训练完了，更新学习率
      if epoch_end:
        self.epoch += 1
        for k in self.optim_configs:
          self.optim_configs[k]['learning_rate'] *= self.lr_decay
      first_it = (t == 0)
      last_it = (t == num_iterations + 1)
      if first_it or last_it or epoch_end:
        train_acc = self.check_accuracy(self.X_train, self.y_train,
                                        num_samples=1000)
        val_acc = self.check_accuracy(self.X_val, self.y_val)
        #if epoch_end:
        #    if abs(val_acc-self.val_acc_history[-1]<0.0008):
        #        for k in self.optim_configs:
        #            self.optim_configs[k]['learning_rate'] *= 0.75
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)

        if self.verbose:
          print '(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                 self.epoch, self.num_epochs, train_acc, val_acc)
      #保存最好的model
        if val_acc > self.best_val_acc:
          self.best_val_acc = val_acc
          self.best_params = {}
          for k, v in self.model.params.iteritems():
            self.best_params[k] = v.copy()

    self.model.params = self.best_params
  def Save(self):
      loss_str = cPickle.dumps(self.loss_history)
      f = open('paramdata/loss_his.txt','wb')
      f.write(loss_str)
      f.close()
      train_acc_str = cPickle.dumps(self.train_acc_history)
      f = open('paramdata/train_acc_his.txt','wb')
      f.write(train_acc_str)
      f.close()
      val_acc_str = cPickle.dumps(self.val_acc_history)
      f = open('paramdata/val_acc_his.txt','wb')
      f.write(val_acc_str)
      f.close()
      best_params_str = cPickle.dumps(self.best_params)
      f = open('paramdata/best_params.txt','wb')
      f.write(best_params_str)
      f.close()
      #保存最好的模型
      f = open('paramdata/model','wb')
      cPickle.dump(self.model,f,protocol=cPickle.HIGHEST_PROTOCOL)
      f.close()
      print 'done'
      
      
      
      
      
      
      