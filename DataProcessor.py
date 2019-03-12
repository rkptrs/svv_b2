# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:47:54 2019

@author: jonas
"""
import numpy as np
import scipy.io
import cPickle as pickle

mat = scipy.io.loadmat('flightdatatest.mat')

with open('test.txt', 'w') as file:
    file.write(pickle.dumps(mat))