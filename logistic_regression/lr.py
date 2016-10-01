# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 01:09:03 2014

@author: atulkumar
"""

from __future__ import division
import numpy as np

def logistic(a): 
    return 1.0 / (1 + np.exp(-a))

def irls(X, y): 
    theta = np.zeros(X.shape[1]) 
    theta_ = np.inf 
    while max(abs(theta-theta_)) > 1e-6: 
        a = np.dot(X, theta) 
        pi = logistic(a) 
        SX = X * (pi - pi*pi).reshape(-1,1) 
        XSX = np.dot(X.T, SX) 
        SXtheta = np.dot(SX, theta) 
        theta_ = theta 
        theta = np.linalg.solve(XSX, np.dot(X.T, SXtheta + y - pi)) 
        return theta
