# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:52:44 2019

@author: jonas
"""
import numpy as np
import matplotlib.pyplot as plt

time = np.loadtxt('time.dat', dtype = 'float')

aoa = np.loadtxt('aoa.dat', dtype = 'float')
#rollangle = np.loadtxt('rollangle.dat', dtype = 'float')
#pitchangle = np.loadtxt('pitchangle.dat', dtype = 'float')

#rollrate = np.loadtxt('rollrate.dat', dtype = 'float')
#pitchrate = np.loadtxt('pitchrate.dat', dtype = 'float')

measrunning = np.loadtxt('measrunning.dat', dtype = 'float')

plt.plot(time, aoa)
plt.plot(time, measrunning)

plt.legend(['AoA', 'Measurement Running', 'Number of Measurement'])

plt.show()