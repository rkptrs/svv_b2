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

def get_plots(paramstr, choosedemo, time, outputdata): #returns plots of data chosen for export in givedata, need string of parameter symbols for legend, string for title, time for x-axis, output data to construct lines on y-axis
    #Plotting the chosen parameters
    for i in range(0, outputdata.shape[1]):
        plt.plot(time, outputdata[:,i])
    
    plt.legend(paramstr)
    plt.xlabel('Time [s]')
    plt.xlim((min(time), max(time)))
    plt.title(choosedemo)
    plt.grid()
    
    plt.show()
    
    return
    
def givedata(choosedemo, chooseparam, plotting): #function that will return flight data based on chosen manoeuvre (choosedemo, string must match entry in "measdemotype") and chosen parameters (chooseparam, list of strings must match entries in "variables"), plots all data into one plot if plotting = True
    #Entering flight data from XLSX
    meastimes1 = ['00:20:10', '00:21:40', '00:23:40', '00:26:30', '00:28:45', '00:31:00'] #times of measurements for series 1
    meastimes2 = ['00:36:00', '00:37:30', '00:39:35', '00:41:10', '00:43:05', '00:44:20', '00:45:25', '00:46:57', '00:48:52'] #times of measurement for series 2
    meas2cgshift = [False, False, False, False, False, False, False, True, True] #for which measurements in series 2 was the cg shifted
    meastimesdemo = ['00:50:07', '00:51:24', '00:52:30', '00:55:53', '00:57:30', '01:00:02', '01:04:00']#measurement times for the demonstration part
    measdemotype = ['Short period', 'Aperiodic Roll', 'Phugoid', 'Dutch Roll', 'Dutch Roll with YD', 'Spiral', 'Parabola'] #which type of manoeuvre was flown

    #Entering available parameters
    variables = ['aoa', 'dte', 'fe', 'lhfmf', 'rhfmf', 'tfulbs', 'tfukg', 'deltaa', 'deltae', 'deltar', 'rollangle', 'pitchangle', 'rollrate', 'pitchrate', 'yawrate', 'tt', 'hp', 'm', 'tas'] #list of available variable names
    param = ['alpha', 'delta_t_e', 'F_e', 'FFl', 'FFr', 'Wf', 'Wf', 'delta_a', 'delta_e', 'delta_r', 'phi', 'theta', 'p', 'q', 'r', 'T_0', 'h', 'M', 'V_TAS'] #list of corresponding symbols
#    param = ['α', 'δte', 'Fe', 'FFl', 'FFr', 'Wf', 'Wf', 'δa', 'δe', 'δr', 'ϕ', 'θ', 'p', 'q', 'r', 'T0', 'h', 'M', 'Vtas'] #list of corresponding symbols
    units = ['[deg]', '[deg]', '[N]', '[kg/s]', '[kg/s]', '[lbs]', '[kg]', '[deg]', '[deg]', '[deg]', '[deg]', '[deg]', '[deg/s]', '[deg/s]', '[deg/s]', '[K]', '[m]', '[-]', '[m/s]'] #list of corresponding units
    
    #Loading all external data
    time = np.loadtxt('time.dat', dtype = 'float')

    aoa = np.loadtxt('aoa.dat', dtype = 'float')
    deltaa = np.loadtxt('deltaa.dat', dtype = 'float')
    deltae = np.loadtxt('deltae.dat', dtype = 'float')
    deltar = np.loadtxt('deltar.dat', dtype = 'float')
    dte = np.loadtxt('dte.dat', dtype = 'float')
    fe = np.loadtxt('fe.dat', dtype = 'float')
    hp = np.loadtxt('hp.dat', dtype = 'float')
    lhfmf = np.loadtxt('lhfmf.dat', dtype = 'float')
    mach = np.loadtxt('mach.dat', dtype = 'float')
    pitchangle = np.loadtxt('pitchangle.dat', dtype = 'float')
    pitchrate = np.loadtxt('pitchrate.dat', dtype = 'float')
    rhfmf = np.loadtxt('rhfmf.dat', dtype = 'float')
    rollangle = np.loadtxt('rollangle.dat', dtype = 'float')
    rollrate = np.loadtxt('rollrate.dat', dtype = 'float')
    tas = np.loadtxt('tas.dat', dtype = 'float')
    tfukg = np.loadtxt('tfukg.dat', dtype = 'float')
    tfulbs = np.loadtxt('tfulbs.dat', dtype = 'float')
    tt = np.loadtxt('tt.dat', dtype = 'float')
    yawrate = np.loadtxt('yawrate.dat', dtype = 'float')

    #Computing desired time
    demoindex = measdemotype.index(choosedemo)
    starttime = get_sec(meastimesdemo[demoindex])
    
    if choosedemo == 'Parabola':
        endtime = starttime + 30
    else:
        endtime = get_sec(meastimesdemo[demoindex + 1]) #if parabola is chosen, there is no time recorded afterwards
        
    #Finding desired time
    starttimeindex = np.where(time == starttime)
    endtimeindex = np.where(time == endtime)
    
    chosentime = time[starttimeindex[0]:endtimeindex[0]] #time array for chosen manoeuvre, 0 is necessary due to nature of output of np.where as an array
    
    #filtering chosen parameters by time span
    outputdata = np.zeros((len(chosentime),len(chooseparam)))
    paramstr = []
    i = 0
    for parameter in chooseparam:
        data = eval(parameter)
        
        outputdata[:,i] = data[starttimeindex[0]:endtimeindex[0]] #0 is necessary due to nature of output of np.where as an array
        
        #building parameter legend labels from chooseparam
        paramindex = variables.index(parameter)
        paramstr.append(param[paramindex] + ' ' + units[paramindex]) #glueing the legend string together with nice label and units      

        i += 1
        
    if plotting:
        get_plots(paramstr, choosedemo, chosentime, outputdata)
        
    return chosentime, outputdata

data = givedata('Phugoid', ['aoa','pitchangle','pitchrate','tas'], True)

""" Putting lines into the full time-scale plots
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
    plt.text(seconds + 20, 0, measdemotype[i], rotation = 90)
    i += 1
"""