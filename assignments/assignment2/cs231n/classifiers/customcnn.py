import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class CustomConvNet(object):

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               conv_layers=1, use_batchnorm=False, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.conv_layers = conv_layers
    self.num_layers = conv_layers + 2 # Currently conv + affine + softmax
    self.use_batchnorm = use_batchnorm

    if self.use_batchnorm:
        self.bn_params = []
        self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers + 1)]

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    F = num_filters
    HH = filter_size
    WW = filter_size

    layer_dim = (F, C, HH, WW)

    # Conv - relu - pool weights
    for l in xrange(1, self.conv_layers + 1):
        self.params['W%d' % l] = np.random.normal(loc=0.0, scale=weight_scale, size=layer_dim)
        self.params['b%d' % l] = np.zeros(F)
        if self.use_batchnorm:
            self.params['gamma%d' % l] = np.ones(F)
            self.params['beta%d' % l] = np.zeros(F)
        layer_dim = (F, F, HH, WW)

    # Affine - Relu layer
    l = self.conv_layers + 1
    h_shape = ((num_filters * np.prod(input_dim[1:]) / 4**self.conv_layers), hidden_dim)
    self.params['W%d' % l] = np.random.normal(loc=0.0, scale=weight_scale, size=h_shape)
    self.params['b%d' % l] = np.zeros(hidden_dim)
    if self.use_batchnorm:
        self.params['gamma%d' % l] = np.ones(hidden_dim)
        self.params['beta%d' % l] = np.zeros(hidden_dim)

    # Final affine layer (hidden layers -> classes)
    l = l + 1
    a_shape = (hidden_dim, num_classes)
    self.params['W%d' % l] = np.random.normal(loc=0.0, scale=weight_scale, size=a_shape)
    self.params['b%d' % l] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def get_params_for_layer(self, layer, get_gamma_beta=False):
    W = self.params['W%d' % layer]
    b = self.params['b%d' % layer]
    if get_gamma_beta:
        gamma = self.params['gamma%d' % layer]
        beta = self.params['beta%d' % layer]
        return W, b, gamma, beta
    return W, b

  def set_grads(self, layer, grads, dw, db, dgamma=None, dbeta=None):
    grads['W%d' % layer] = dw
    grads['b%d' % layer] = db
    if dgamma is not None and dbeta is not None:
      grads['gamma%d' % layer] = dgamma
      grads['beta%d' % layer] = dbeta

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1 = self.params['W1']
    mode = 'test' if y is None else 'train'

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    cache = {}

    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    input = X
    for l in xrange(1, self.conv_layers + 1):
        if self.use_batchnorm:
            W, b, gamma, beta = self.get_params_for_layer(l, get_gamma_beta=True)
            input, cache['cache%d' % l] = conv_norm_relu_pool_forward(input, W, b, conv_param, pool_param, gamma, beta, self.bn_params[l])
        else:
            W, b = self.get_params_for_layer(l)
            input, cache['cache%d' % l] = conv_relu_pool_forward(input, W, b, conv_param, pool_param)

    l = self.conv_layers + 1
    if self.use_batchnorm:
        W, b, gamma, beta = self.get_params_for_layer(l, get_gamma_beta=True)
        h_out, h_cache = affine_norm_relu_forward(input, W, b, gamma, beta, self.bn_params[l])
    else:
        W, b = self.get_params_for_layer(l)
        h_out, h_cache = affine_relu_forward(input, W, b)

    l = l + 1
    W, b = self.get_params_for_layer(l)
    scores, scores_cache = affine_forward(h_out, W, b)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, loss_dx = softmax_loss(scores, y)

    for l in xrange(1, self.num_layers + 1):
        loss += 0.5 * self.reg * np.sum(self.params['W%d' % l] * self.params['W%d' % l])

    l = self.num_layers
    scores_dx, scores_dw, scores_db = affine_backward(loss_dx, scores_cache)
    self.set_grads(l, grads, scores_dw, scores_db)
    l = l - 1

    if self.use_batchnorm:
      a_dx, a_dw, a_db, a_dgamma, a_dbeta = affine_norm_relu_backward(scores_dx, h_cache)
      self.set_grads(l, grads, a_dw, a_db, a_dgamma, a_dbeta)
    else:
      a_dx, a_dw, a_db = affine_relu_backward(scores_dx, h_cache)
      self.set_grads(l, grads, a_dw, a_db)
    l = l - 1

    conv_layers = l
    next_input = a_dx
    for l in xrange(conv_layers, 0, -1):
        current_cache = cache['cache%d' % l]
        if self.use_batchnorm:
          c_dx, c_dw, c_db, c_dgamma, c_dbeta = conv_norm_relu_pool_backward(next_input, current_cache)
          self.set_grads(l, grads, c_dw, c_db, c_dgamma, c_dbeta)
        else:
          c_dx, c_dw, c_db = conv_relu_pool_backward(next_input, current_cache)
          self.set_grads(l, grads, c_dw, c_db)
        next_input = c_dx

    for l in xrange(1, self.conv_layers + 3):
        grads['W%d' % l] += self.reg * self.params['W%d' % l]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
