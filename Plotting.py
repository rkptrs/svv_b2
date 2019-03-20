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
    meastimesdemo = ['00:50:07', '00:51:24', '00:52:34', '00:55:53', '00:57:30', '01:00:02', '01:04:00']#measurement times for the demonstration part
    measdemotype = ['Short period', 'Aperiodic Roll', 'Phugoid', 'Dutch Roll', 'Dutch Roll with YD', 'Spiral', 'Parabola'] #which type of manoeuvre was flown

    #Entering available parameters
    variables = ['aoa', 'dte', 'fe', 'lhfmf', 'rhfmf', 'tfulbs', 'tfukg', 'deltaa', 'deltae', 'deltar', 'rollangle', 'pitchangle', 'rollrate', 'pitchrate', 'yawrate', 'tt', 'hp', 'm', 'tas'] #list of available variable names
    param = ['alpha', 'delta_t_e', 'F_e', 'FFl', 'FFr', 'Wf', 'Wf', 'delta_a', 'delta_e', 'delta_r', 'phi', 'theta', 'p', 'q', 'r', 'T_0', 'h', 'M', 'V_TAS'] #list of corresponding symbols
#    param = ['α', 'δte', 'Fe', 'FFl', 'FFr', 'Wf', 'Wf', 'δa', 'δe', 'δr', 'ϕ', 'θ', 'p', 'q', 'r', 'T0', 'h', 'M', 'Vtas'] #list of corresponding symbols
    units = ['[deg]', '[deg]', '[N]', '[kg/s]', '[kg/s]', '[lbs]', '[kg]', '[deg]', '[deg]', '[deg]', '[deg]', '[deg]', '[deg/s]', '[deg/s]', '[deg/s]', '[K]', '[m]', '[-]', '[m/s]'] #list of corresponding units
    
    #Loading all external data
    time = np.loadtxt('FlightData/time.dat', dtype = 'float')

    aoa = np.loadtxt('FlightData/aoa.dat', dtype = 'float')
    deltaa = np.loadtxt('FlightData/deltaa.dat', dtype = 'float')
    deltae = np.loadtxt('FlightData/deltae.dat', dtype = 'float')
    deltar = np.loadtxt('FlightData/deltar.dat', dtype = 'float')
    dte = np.loadtxt('FlightData/dte.dat', dtype = 'float')
    fe = np.loadtxt('FlightData/fe.dat', dtype = 'float')
    hp = np.loadtxt('FlightData/hp.dat', dtype = 'float')
    lhfmf = np.loadtxt('FlightData/lhfmf.dat', dtype = 'float')
    mach = np.loadtxt('FlightData/mach.dat', dtype = 'float')
    pitchangle = np.loadtxt('FlightData/pitchangle.dat', dtype = 'float')
    pitchrate = np.loadtxt('FlightData/pitchrate.dat', dtype = 'float')
    rhfmf = np.loadtxt('FlightData/rhfmf.dat', dtype = 'float')
    rollangle = np.loadtxt('FlightData/rollangle.dat', dtype = 'float')
    rollrate = np.loadtxt('FlightData/rollrate.dat', dtype = 'float')
    tas = np.loadtxt('FlightData/tas.dat', dtype = 'float')
    tfukg = np.loadtxt('FlightData/tfukg.dat', dtype = 'float')
    tfulbs = np.loadtxt('FlightData/tfulbs.dat', dtype = 'float')
    tt = np.loadtxt('FlightData/tt.dat', dtype = 'float')
    yawrate = np.loadtxt('FlightData/yawrate.dat', dtype = 'float')

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

    chosentime = time[starttimeindex[0][0]:endtimeindex[0][0]] #time array for chosen manoeuvre, 0 is necessary due to nature of output of np.where as an array
    
    #filtering chosen parameters by time span
    outputdata = np.zeros((len(chosentime),len(chooseparam)))
    paramstr = []
    i = 0
    for parameter in chooseparam:
        data = eval(parameter)
        
        outputdata[:,i] = data[starttimeindex[0][0]:endtimeindex[0][0]] #0 is necessary due to nature of output of np.where as an array
        
        #building parameter legend labels from chooseparam
        paramindex = variables.index(parameter)
        paramstr.append(param[paramindex] + ' ' + units[paramindex]) #glueing the legend string together with nice label and units      

        i += 1
        
    if plotting:
        get_plots(paramstr, choosedemo, chosentime, outputdata)
        
    return chosentime, outputdata

def compare_aperiodic(modeldata, flightdata): #plots comparison plots for aperiodic roll, needs data arrays from model and flight with columns for time, beta (only simulation, zeros/ones for flight data), phi, p, r, delta_r, delta_a
    plt.subplot(511) #beta
    plt.plot(modeldata[:,0], modeldata[:,1]) #model data
    plt.xlabel('Time [s]')
    plt.ylabel('β [°]')
    plt.legend(['Simulation'])
    plt.grid()
    
    plt.subplot(512) #phi
    plt.plot(modeldata[:,0], modeldata[:,2]) #model data
    plt.plot(flightdata[:,0], flightdata[:,2]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('ϕ [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(513) #p
    plt.plot(modeldata[:,0], modeldata[:,3]) #model data
    plt.plot(flightdata[:,0], flightdata[:,3]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('p [°/s]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(514) #r
    plt.plot(modeldata[:,0], modeldata[:,4]) #model data
    plt.plot(flightdata[:,0], flightdata[:,4]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('r [°/s]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(515) #deltar, deltaa
    plt.plot(modeldata[:,0], modeldata[:,5]) #model data
    plt.plot(flightdata[:,0], flightdata[:,5]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('δr [°], δa [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.show()
        
    return

def compare_dutchroll(modeldata, flightdata): #plots comparison plots for dutch roll, needs data arrays from model and flight with columns for time, beta (only simulation, zeros/ones for flight data), phi, p, r, delta_r, delta_a
    plt.subplot(511) #beta
    plt.plot(modeldata[:,0], modeldata[:,1]) #model data
    plt.xlabel('Time [s]')
    plt.ylabel('β [°]')
    plt.legend(['Simulation'])
    plt.grid()
    
    plt.subplot(512) #phi
    plt.plot(modeldata[:,0], modeldata[:,2]) #model data
    plt.plot(flightdata[:,0], flightdata[:,2]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('ϕ [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(513) #p
    plt.plot(modeldata[:,0], modeldata[:,3]) #model data
    plt.plot(flightdata[:,0], flightdata[:,3]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('p [°/s]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(514) #r
    plt.plot(modeldata[:,0], modeldata[:,4]) #model data
    plt.plot(flightdata[:,0], flightdata[:,4]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('r [°/s]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(515) #deltar, deltaa
    plt.plot(modeldata[:,0], modeldata[:,5]) #model data
    plt.plot(flightdata[:,0], flightdata[:,5]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('δr [°], δa [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.show()
        
    return

def compare_phugoid(modeldata, flightdata): #plots comparison plots for phugoid, needs data arrays from model and flight with columns for time, V, alpha, theta, q, delta_e
    #fig_phugoid = plt.figure()
    plt.subplot(511) #V
    plt.plot(modeldata[:,0], modeldata[:,1]) #model data
    plt.plot(flightdata[:,0], flightdata[:,1]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('V [m/s]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(512) #alpha
    plt.plot(modeldata[:,0], modeldata[:,2]) #model data
    plt.plot(flightdata[:,0], flightdata[:,2]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('α [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(513) #theta
    plt.plot(modeldata[:,0], modeldata[:,3]) #model data
    plt.plot(flightdata[:,0], flightdata[:,3]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('θ [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(514) #q
    plt.plot(modeldata[:,0], modeldata[:,4]) #model data
    plt.plot(flightdata[:,0], flightdata[:,4]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('q [°/s]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(515) #deltae
    plt.plot(modeldata[:,0], modeldata[:,5]) #model data
    plt.plot(flightdata[:,0], flightdata[:,5]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('δe [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.show()
    
    return

def compare_shortperiod(modeldata, flightdata): #plots comparison plots for aperiodic roll, needs data arrays from model and flight with columns for time, V, alpha, theta, q, delta_e
    plt.subplot(511) #V
    plt.plot(modeldata[:,0], modeldata[:,1]) #model data
    plt.plot(flightdata[:,0], flightdata[:,1]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('V [m/s]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(512) #alpha
    plt.plot(modeldata[:,0], modeldata[:,2]) #model data
    plt.plot(flightdata[:,0], flightdata[:,2]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('α [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(513) #theta
    plt.plot(modeldata[:,0], modeldata[:,3]) #model data
    plt.plot(flightdata[:,0], flightdata[:,3]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('θ [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(514) #q
    plt.plot(modeldata[:,0], modeldata[:,4]) #model data
    plt.plot(flightdata[:,0], flightdata[:,4]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('q [°/s]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(515) #deltae
    plt.plot(modeldata[:,0], modeldata[:,5]) #model data
    plt.plot(flightdata[:,0], flightdata[:,5]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('δe [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.show()
        
    return

def compare_spiral(modeldata, flightdata): #plots comparison plots for aperiodic roll, needs data arrays from model and flight with columns for time, beta (only simulation, zeros/ones for flight data), phi, p, r, delta_r, delta_a
    plt.subplot(511) #beta
    plt.plot(modeldata[:,0], modeldata[:,1]) #model data
    plt.xlabel('Time [s]')
    plt.ylabel('β [°]')
    plt.legend(['Simulation'])
    plt.grid()
    
    plt.subplot(512) #phi
    plt.plot(modeldata[:,0], modeldata[:,2]) #model data
    plt.plot(flightdata[:,0], flightdata[:,2]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('ϕ [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(513) #p
    plt.plot(modeldata[:,0], modeldata[:,3]) #model data
    plt.plot(flightdata[:,0], flightdata[:,3]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('p [°/s]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(514) #r
    plt.plot(modeldata[:,0], modeldata[:,4]) #model data
    plt.plot(flightdata[:,0], flightdata[:,4]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('r [°/s]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.subplot(515) #deltar, deltaa
    plt.plot(modeldata[:,0], modeldata[:,5]) #model data
    plt.plot(flightdata[:,0], flightdata[:,5]) #flight data
    plt.xlabel('Time [s]')
    plt.ylabel('δr [°], δa [°]')
    plt.legend(['Simulation', 'Flight Data'])
    plt.grid()
    
    plt.show()
        
    return

#time, data = givedata('Phugoid', ['aoa','pitchangle','pitchrate','tas'], True)

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
