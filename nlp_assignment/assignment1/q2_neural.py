#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    #raise NotImplementedError
    N = labels.shape[0]
    h1 = np.dot(data, W1) #M x Dx , Dx, H  => M x H
    o1 = h1 + b1 #M x H
    sig1 = sigmoid(o1) #M x H
    #sig1 = np.maximum(0, o1)
    h2 = np.dot(sig1, W2) #M x H, H, Dy => M x Dy
    o2 = h2 + b2 #M x Dy

    y_hat = softmax(o2)
    '''
    scores = o2 #M x Dy
    scores -= np.max(scores, axis = 1).reshape(-1, 1)
    y_hat = np.exp(scores)/np.sum(np.exp(scores), axis = 1).reshape(-1, 1)
    '''
    cost = -np.sum(np.multiply(np.log(y_hat), labels))/N

    #backprop
    dy_hat = y_hat
    dy_hat -= labels
    dy_hat = dy_hat/N
    do2 = dy_hat

    gradb2 = np.sum(do2, axis=0)
    dh2 = do2
    gradW2 = np.dot(sig1.T, dh2)
    dsig1 = np.dot(dh2, W2.T)
    #dsig1[sig1 <= 0] = 0
    #do1 = dsig1
    do1 = (sig1*(1-sig1))*dsig1

    gradb1 = np.sum(do1, axis=0)
    dh1 = do1
    gradW1 = np.dot(data.T, dh1)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()
