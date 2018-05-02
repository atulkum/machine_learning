import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C = W.shape[1]
  N = X.shape[0]
  for i in xrange(N):
    h = X[i].dot(W) 
    h -= np.max(h)
    p = np.exp(h) / np.sum(np.exp(h))
    
    loss += -np.log(p[y[i]])

    p[y[i]] -= 1 
    
    for j in xrange(C):
      dW[:,j] += X[i,:] * p[j]

  loss /= N
  dW /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  h = X.dot(W)
  h -= np.max(h, axis = 1).reshape(-1, 1)
  probs = np.exp(h)/np.sum(np.exp(h), axis = 1).reshape(-1, 1)

  loss = -(np.sum(np.log(probs[np.arange(len(y)), y])))/N
  loss += 0.5*reg*np.sum(W.dot(W.T))
 
  probs[range(len(y)),y] -= 1
  dW = X.T.dot(probs) / N
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

