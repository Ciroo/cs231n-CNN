from classifiers.cnn import *
from load_data import *
from layers.layers import *
from layers.layer_utils import *
from solver.solver import *
from time import *

def check_accuracy(X, y, model, num_samples=None, batch_size=100):

   N = X.shape[0]
   if num_samples is not None and N > num_samples:
      mask = np.random.choice(N, num_samples)
      N = num_samples
      X = X[mask]
      y = y[mask]

   num_batches = N / batch_size
   if N % batch_size != 0:
      num_batches += 1
   y_pred = []
   for i in xrange(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      scores = model.loss(X[start:end])
      y_pred.append(np.argmax(scores, axis=1))
   y_pred = np.hstack(y_pred)
   acc = np.mean(y_pred == y)

   return acc
if __name__ == "__main__":
   data = get_MNIST_data()
   f = open('paramdata/model','rb')
   MyModel = cPickle.load(f)
   print check_accuracy(data['X_test'], data['y_test'],MyModel)
   f.close()