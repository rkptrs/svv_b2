# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:52:44 2019

@author: jonas
"""
import numpy as np
import matplotlib.pyplot as plt

def get_sec(time_str): #takes as input a timestamp string of format (H)H:MM:SS, returns equivalent number of seconds
    h, m, s = time_str.split(':') 
    
    return int(h) * 3600 + int(m) * 60 + int(s)
    
def plottingattime(starttime, x, y):
    
    
    plt.show()
    return

meastimes1 = ['00:20:10', '00:21:40', '00:23:40', '00:26:30', '00:28:45', '00:31:00'] #times of measurements for series 1
meastimes2 = ['00:36:00', '00:37:30', '00:39:35', '00:41:10', '00:43:05', '00:44:20', '00:45:25', '00:46:57', '00:48:52'] #times of measurement for series 2
meas2cgshift = [False, False, False, False, False, False, False, True, True] #for which measurements in series 2 was the cg shifted
meastimesdemo = ['00:50:07', '00:51:24', '00:52:30', '00:55:53', '00:57:30', '01:00:02', '01:04:00']#measurement times for the demonstration part
measdemotype = ['Short period', 'Aperiodic Roll', 'Phugoid', 'Dutch Roll', 'Dutch Roll with YD', 'Spiral', 'Parabola'] #which type of manoeuvre was flown

time = np.loadtxt('time.dat', dtype = 'float')

#aoa = np.loadtxt('aoa.dat', dtype = 'float')
rollangle = np.loadtxt('rollangle.dat', dtype = 'float')
#pitchangle = np.loadtxt('pitchangle.dat', dtype = 'float')

rollrate = np.loadtxt('rollrate.dat', dtype = 'float')
#pitchrate = np.loadtxt('pitchrate.dat', dtype = 'float')

#measrunning = np.loadtxt('measrunning.dat', dtype = 'float')

#plt.plot(time, aoa)
#plt.plot(time, measrunning)
plt.plot(time, rollangle)
plt.plot(time, rollrate)

plt.legend(['Roll Angle [deg]', 'Roll Rate [deg/s]']) #'AoA', 'Measurement Running', 

#adding timestamps of series 1 in black
for stamp in meastimes1:
    seconds = get_sec(stamp)
    plt.axvline(seconds, color = 'k')
    
#adding timestamps of series 2 in yellow
for stamp in meastimes2:
    seconds = get_sec(stamp)
    plt.axvline(seconds, color = 'y')
    
#adding timestamps of demonstration section in red with labels of manoeuvre
i = 0 #index counter
for stamp in meastimesdemo:
    seconds = get_sec(stamp)
    plt.axvline(seconds, color = 'r')
    plt.text(seconds + 20, 50, measdemotype[i], rotation = 90)
    i += 1

plt.show()