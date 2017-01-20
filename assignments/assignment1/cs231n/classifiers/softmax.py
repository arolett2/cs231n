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

  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_dims = W.shape[0]

  for i in xrange(num_train):
    raw_scores = X[i].dot(W)
    scores = raw_scores - np.max(raw_scores)

    loss -= scores[y[i]]

    sum_exp = 0.0
    for s in scores:
      sum_exp += np.exp(s)

    for c in xrange(num_classes):
      p = np.exp(scores[c]) / sum_exp
      dW[:, c] += -((c == y[i])-p) * X[i, :]

    loss += np.log(sum_exp)

  loss /= num_train
  dW /= num_train

  dW += reg * W

  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  raw_scores = X.dot(W)
  scores = raw_scores - np.max(raw_scores, axis=1).reshape(num_train, -1)

  loss -= np.sum(scores[np.arange(0, scores.shape[0]), y].reshape(num_train, -1))

  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores, axis=1).reshape(num_train, -1)

  probs = exp_scores / sum_exp_scores
  one_hot = np.eye(num_classes)[y]

  dClass = one_hot - probs
  dClass *= -1

  dW = X.T.dot(dClass)

  loss += np.sum(np.log(sum_exp_scores))

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

