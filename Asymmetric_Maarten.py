import control.matlab as cmat
import control
import numpy as np
import matplotlib.pyplot as plt
from Cit_para import *

#define initial paramaters

V = V0

    #create C1 C2 C3
C1 = [[(CYbdot-2*mub)*b/V, 0, 0, 0],[0, 1, 0, 0],[0, 0, -2*mub*KX2*b**2/V**2, 2*mub*KXZ*b**2/V**2],[Cnbdot*b/V, 0, 2*mub*KXZ*b**2/V**2, -2*mub*KZ2*b**2/V**2]]
C2 = [[CYb, CL, CYp*b/(2*V), (CYr-4*mub)*b/(2*V)],[0, 0, -1, 0],[Clb, 0, Clp*b/(2*V), Clr*b/(2*V)], [Cnb, 0, Cnp*b/(2*V), Cnr*b/(2*V)]]
C3 = [[CYda, CYdr], [0, 0], [Clda, Cldr],[Cnda, Cndr]]

C1inv = np.linalg.inv(C1)
A_asym2 = np.matmul(-C1inv,C2)
B_asym2 = np.matmul(-C1inv,C3)
C_asym2 = np.identity(4)
D_asym2 = np.zeros((4,2))

#print(np.linalg.eigvals(A_asym2)*b/V)

#eigenvalues
#eigv = np.linalg.eigvals(A_asym)
#eigv2 = np.linalg.eigvals(A_asym2)
#print(eigv- eigv2)



#read flight data
time = np.loadtxt('FlightData/time.dat', dtype = 'float')
deltaa = np.loadtxt('FlightData/deltaa.dat', dtype = 'float')
deltar = np.loadtxt('FlightData/deltar.dat', dtype = 'float')
deltae = np.loadtxt('FlightData/deltae.dat', dtype = 'float')

yawrate = np.loadtxt('FlightData/yawrate.dat', dtype = 'float')
rollangle = np.loadtxt('FlightData/rollangle.dat', dtype = 'float')
rollrate = np.loadtxt('FlightData/rollrate.dat', dtype = 'float')
pitchangle = np.loadtxt('FlightData/pitchangle.dat', dtype = 'float')
tas = np.loadtxt('FlightData/tas.dat', dtype = 'float')
hp = np.loadtxt('FlightData/hp.dat', dtype = 'float')
fuelused = np.loadtxt('FlightData/tfulbs.dat', dtype = 'float')

meastimesdemo = ['00:50:07', '00:51:13', '00:52:30', '00:55:42', '00:57:18', '00:59:43', '01:04:00']#measurement times for the demonstration part
measdemotype = ['Short period', 'Aperiodic Roll', 'Phugoid', 'Dutch Roll', 'Dutch Roll with YD', 'Spiral', 'Parabola'] #which type of manoeuvre was flown

def get_sec(time_str): #takes as input a timestamp string of format (H)H:MM:SS, returns equivalent number of seconds
    h, m, s = time_str.split(':')

    return int(h) * 3600 + int(m) * 60 + int(s)
meastimesdemosecs = np.zeros(7)

for i in range(len(meastimesdemo)):
    meastimesdemosecs[i] = get_sec(meastimesdemo[i])*10

##set motion
#demo = [3]
##state-space
##output [1,2,3,4] = [beta, phi, p, r]
##range aperiodic roll = index 30740 for 20 seconds
#
#
##time period of response; 1300 for spiral, 200 for rest
#for i in demo:
#    if i == 1 or i ==3:
#        Trng = 200
#    elif i == 5:
#        Trng = 1300
#rng = [int(meastimesdemosecs[demo]),int(meastimesdemosecs[demo])+Trng]
#T = np.linspace(0,Trng/10,Trng)
#
#Unew = np.zeros((Trng,2))
#for i in range(Trng):
#    Unew[i][1] = deltar[rng[0]+i]
#    Unew[i][0] = deltaa[rng[0]+i]
#
#sys2 = cmat.ss(A_asym2, B_asym2, C_asym2, D_asym2)
#response = cmat.lsim(sys2, T=T, U=-Unew, X0 = [0,rollangle[rng[0]],rollrate[rng[0]],yawrate[rng[0]]])
##testroll = cmat.lsim(sys2, T=T, U=-Unew, X0 = [0,rollangle[rng[0]],rollrate[rng[0]],yawrate[rng[0]]])





def asym_err():
    error = []
    for i in range(len(meastimesdemo)):
        meastimesdemosecs[i] = get_sec(meastimesdemo[i])*10
    errloop = []
    demo = [1,3,5]
    for i in demo:
    #time period of response; 1300 for spiral, 200 for rest
        if i == 1 or i == 3:
            Trng = 200
        else:
            Trng = 1300
        rng = [int(meastimesdemosecs[i]),int(meastimesdemosecs[i])+Trng]
        
        T = np.linspace(0,Trng/10,Trng)
    
        Unew = np.zeros((Trng,2))
    
        for n in range(Trng):
            Unew[n][1] = deltar[rng[0]+n]
            Unew[n][0] = deltaa[rng[0]+n]
            
        minit = 14064.8 #lbs
        mused = fuelused[rng[0]]
        hp0 = hp[rng[0]]
        V0 = tas[rng[0]]
        V = V0
        m = (minit-mused)*0.45359237
        rho    = rho0 * pow( ((1+(Lambda * hp0 / Temp0))), (-((g / (Lambda*R)) + 1)))  
        W      = m * g    
        CL = 2 * W / (rho * V0 ** 2 * S)              # Lift coefficient [ ]
        CD = CD0 + (CLa * alpha0) ** 2 / (pi * A * e) # Drag coefficient [ ]
        mub    = m / (rho * S * b)
        
            
            #create C1 C2 C3
        C1 = [[(CYbdot-2*mub)*b/V, 0, 0, 0],[0, 1, 0, 0],[0, 0, -2*mub*KX2*b**2/V**2, 2*mub*KXZ*b**2/V**2],[Cnbdot*b/V, 0, 2*mub*KXZ*b**2/V**2, -2*mub*KZ2*b**2/V**2]]
        C2 = [[CYb, CL, CYp*b/(2*V), (CYr-4*mub)*b/(2*V)],[0, 0, -1, 0],[Clb, 0, Clp*b/(2*V), Clr*b/(2*V)], [Cnb, 0, Cnp*b/(2*V), Cnr*b/(2*V)]]
        C3 = [[CYda, CYdr], [0, 0], [Clda, Cldr],[Cnda, Cndr]]
        
        C1inv = np.linalg.inv(C1)
        
        A_asym2 = np.matmul(-C1inv,C2)
        B_asym2 = np.matmul(-C1inv,C3)
        C_asym2 = np.identity(4)
        D_asym2 = np.zeros((4,2))

    
        sys2 = cmat.ss(A_asym2, B_asym2, C_asym2, D_asym2)
    
        response = cmat.lsim(sys2, T=T, U=-Unew, X0 = [0,rollangle[rng[0]],rollrate[rng[0]],yawrate[rng[0]]])
        
        
        err = np.sum((np.transpose((response[0][:,3]))-yawrate[rng[0]:rng[1]])**2)+np.sum((np.transpose((response[0][:,1]))-rollangle[rng[0]:rng[1]])**2) + np.sum((np.transpose((response[0][:,2]))-rollrate[rng[0]:rng[1]])**2)
        if i == 1 or i == 3:
            errloop.append(err)
        elif i == 5:
            errloop.append(err*2/13)
    error.append(np.sum(errloop))
    return error



#motion optimazation
#values1 = [[10406.339816487116], -0.11580000000000007, -0.8509999999999994, 0.21447999899999998, 0.11225999900000005, -0.06769900000000008, -0.19499900000000014, -0.7525, -0.5454, 0.33449900000000005]
values1 = [[10894.214922489804], -0.11580000000000004, -0.8509999999999994, 0.22447999899999999, 0.11225999900000003, -0.07269900000000008, -0.19499900000000014, -0.7500, -0.0304, 0.8495]
Clb = values1[1]
Clp = values1[2]
Clr = values1[3]
Cnb = values1[4]
Cnp = values1[5]
Cnr = values1[6]
CYb = values1[7]
CYp = values1[8]
CYr = values1[9]


#rs = [0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.005] 
#df = [0.05, 0.05, 0.025, 0.025, 0.025, 0.005, 0.0025]
#error = [10**10., 0]
#h = 4
#error1 = []
#while h <= 5:
#    clprange = np.arange(Clp-rs[h],Clp+rs[h],df[h])
#    clbrange = np.arange(Clb-rs[h], Clb+rs[h], df[h])
#    clrrange = np.arange(Clr -rs[h],Clr+rs[h],df[h])
#    cnrrange = np.arange(Cnr-rs[h],Cnr+rs[h],df[h])
#    cnbrange = np.arange(Cnb-rs[h],Cnb+rs[h],df[h])
#    cnprange = np.arange(Cnp-rs[h],Cnp+rs[h],df[h])
#    cybrange = np.arange(CYb -rs[h], CYb+rs[h], df[h])
#    cyprange = np.arange(CYp - rs[h], CYp+rs[h],df[h])
#    cyrrange = np.arange(CYr - rs[h], CYr+rs[h],df[h])
#    
#    for i in range(len(clprange)):
#        print(i)
#        for j in range(len(clbrange)):
#            for k in range(len(clrrange)):
#                for l in range(len(cnrrange)):
#                    for n in range(len(cnbrange)):
##    for m in range(len(cybrange)):
##        print(m)
##        for o in range(len(cyprange)):
##            for p in range(len(cyrrange)):
#                        for q in range(len(cnprange)):
#                            Clp = clprange[i]
#                            Clb = clbrange[j]
#                            Clr = clrrange[k]
#                            Cnr = cnrrange[l]
#                            Cnb = cnbrange[n]
#                            Cnp = cnprange[q]
##                            CYb = cybrange[m]
##                            CYp = cyprange[o]
##                            CYr = cyrrange[p]
#                            error1.append([asym_err(), Clp, Clb, Clr, Cnr, Cnb,Cnp, CYb, CYp, CYr])
#
#    error2 = []
#    for i in range(len(error1)):
#        error2.append(error1[i][0])
#    indexe = error2.index(min(error2))
#    h = h + 1
#    Clp = error1[indexe][1]
#    Clb = error1[indexe][2]
#    Clr = error1[indexe][3]
#    Cnr = error1[indexe][4]
#    Cnb = error1[indexe][5]
#    Cnp = error1[indexe][6]
#    CYb = error1[indexe][7]
#    CYp = error1[indexe][8]
#    CYr = error1[indexe][9]
#    values = [min(error2), Clb, Clp, Clr, Cnb, Cnp, Cnr, CYb, CYp, CYr]
#    
#    
#print(values)
#
#



#plotting aperiodic roll
demo = [1]
#state-space
#output [1,2,3,4] = [beta, phi, p, r]
#range aperiodic roll = index 30740 for 20 seconds


#time period of response; 1300 for spiral, 200 for rest
for i in demo:
    if i == 1 or i ==3:
        Trng = 200
    elif i == 5:
        Trng = 1300
rng = [int(meastimesdemosecs[demo]),int(meastimesdemosecs[demo])+Trng]
T = np.linspace(0,Trng/10,Trng) 

minit = 14064.8 #lbs
mused = fuelused[rng[0]]
hp0 = hp[rng[0]]
V0 = tas[rng[0]]
V = V0
m = (minit-mused)*0.45359237
rho    = rho0 * pow( ((1+(Lambda * hp0 / Temp0))), (-((g / (Lambda*R)) + 1)))  
W      = m * g    
CL = 2 * W / (rho * V0 ** 2 * S)              # Lift coefficient [ ]
CD = CD0 + (CLa * alpha0) ** 2 / (pi * A * e) # Drag coefficient [ ]
mub    = m / (rho * S * b)

mub_aperiodic = mub

C1 = [[(CYbdot-2*mub)*b/V, 0, 0, 0],[0, 1, 0, 0],[0, 0, -2*mub*KX2*b**2/V**2, 2*mub*KXZ*b**2/V**2],[Cnbdot*b/V, 0, 2*mub*KXZ*b**2/V**2, -2*mub*KZ2*b**2/V**2]]
C2 = [[CYb, CL, CYp*b/(2*V), (CYr-4*mub)*b/(2*V)],[0, 0, -1, 0],[Clb, 0, Clp*b/(2*V), Clr*b/(2*V)], [Cnb, 0, Cnp*b/(2*V), Cnr*b/(2*V)]]
C3 = [[CYda, CYdr], [0, 0], [Clda, Cldr],[Cnda, Cndr]]

C1inv = np.linalg.inv(C1)
A_asym2 = np.matmul(-C1inv,C2)
B_asym2 = np.matmul(-C1inv,C3)
C_asym2 = np.identity(4)
D_asym2 = np.zeros((4,2))

Aa_aperiodic = A_asym2

Unew = np.zeros((Trng,2))
for i in range(Trng):
    Unew[i][1] = deltar[rng[0]+i]
    Unew[i][0] = deltaa[rng[0]+i]

sys2 = cmat.ss(A_asym2, B_asym2, C_asym2, D_asym2)
response = cmat.lsim(sys2, T=T, U=-Unew, X0 = [0,rollangle[rng[0]],rollrate[rng[0]],yawrate[rng[0]]])

#plot rollangle
f, (ax1,ax2,ax3, ax4) = plt.subplots(4)
f.suptitle('Aperiodic roll motion', fontsize=16)
ax1.plot(T,np.transpose(response[0][:,1]))
ax1.plot(T,rollangle[rng[0]:rng[1]])
ax1.set_ylabel(r'rollangle ($\phi$)')
ax1.set_xlabel('Time (t)')


#plot rollrate

ax2.plot(T,np.transpose(response[0][:,2]))
ax2.plot(T,rollrate[rng[0]:rng[1]])
ax2.set_ylabel('rollrate (p)')
ax2.set_xlabel('Time (t)')


#plot yawrate

ax3.plot(T,np.transpose(response[0][:,3]))
ax3.plot(T,yawrate[rng[0]:rng[1]])
ax3.set_ylabel('yawrate (r)')
ax3.set_xlabel('Time (t)')

ax4.plot(T,np.transpose(response[0][:,0]))
ax4.set_ylabel('sideslip (beta)')
ax4.set_xlabel('Time (t)')


#plotting dutch roll
demo = [3]
#state-space
#output [1,2,3,4] = [beta, phi, p, r]
#range aperiodic roll = index 30740 for 20 seconds


#time period of response; 1300 for spiral, 200 for rest
for i in demo:
    if i == 1 or i ==3:
        Trng = 200
    elif i == 5:
        Trng = 1300
rng = [int(meastimesdemosecs[demo]),int(meastimesdemosecs[demo])+Trng]
T = np.linspace(0,Trng/10,Trng) 

minit = 14064.8 #lbs
mused = fuelused[rng[0]]
hp0 = hp[rng[0]]
V0 = tas[rng[0]]
V = V0
m = (minit-mused)*0.45359237
rho    = rho0 * pow( ((1+(Lambda * hp0 / Temp0))), (-((g / (Lambda*R)) + 1)))  
W      = m * g    
CL = 2 * W / (rho * V0 ** 2 * S)              # Lift coefficient [ ]
CD = CD0 + (CLa * alpha0) ** 2 / (pi * A * e) # Drag coefficient [ ]
mub    = m / (rho * S * b)

mub_dutchroll = mub

C1 = [[(CYbdot-2*mub)*b/V, 0, 0, 0],[0, 1, 0, 0],[0, 0, -2*mub*KX2*b**2/V**2, 2*mub*KXZ*b**2/V**2],[Cnbdot*b/V, 0, 2*mub*KXZ*b**2/V**2, -2*mub*KZ2*b**2/V**2]]
C2 = [[CYb, CL, CYp*b/(2*V), (CYr-4*mub)*b/(2*V)],[0, 0, -1, 0],[Clb, 0, Clp*b/(2*V), Clr*b/(2*V)], [Cnb, 0, Cnp*b/(2*V), Cnr*b/(2*V)]]
C3 = [[CYda, CYdr], [0, 0], [Clda, Cldr],[Cnda, Cndr]]

C1inv = np.linalg.inv(C1)
A_asym2 = np.matmul(-C1inv,C2)
B_asym2 = np.matmul(-C1inv,C3)
C_asym2 = np.identity(4)
D_asym2 = np.zeros((4,2))

Aa_dutchroll = A_asym2

Unew = np.zeros((Trng,2))
for i in range(Trng):
    Unew[i][1] = deltar[rng[0]+i]
    Unew[i][0] = deltaa[rng[0]+i]

sys2 = cmat.ss(A_asym2, B_asym2, C_asym2, D_asym2)
response = cmat.lsim(sys2, T=T, U=-Unew, X0 = [0,rollangle[rng[0]],rollrate[rng[0]],yawrate[rng[0]]])

#plot rollangle
f, (ax1,ax2,ax3) = plt.subplots(3)
f.suptitle('Dutch roll motion', fontsize=16)
ax1.plot(T,np.transpose(response[0][:,1]))
ax1.plot(T,rollangle[rng[0]:rng[1]])
ax1.set_ylabel(r'rollangle ($\phi$)')
ax1.set_xlabel('Time (t)')


#plot rollrate

ax2.plot(T,np.transpose(response[0][:,2]))
ax2.plot(T,rollrate[rng[0]:rng[1]])
ax2.set_ylabel('rollrate (p)')
ax2.set_xlabel('Time (t)')


#plot yawrate

ax3.plot(T,np.transpose(response[0][:,3]))
ax3.plot(T,yawrate[rng[0]:rng[1]])
ax3.set_ylabel('yawrate (r)')
ax3.set_xlabel('Time (t)')


#plotting spiral
demo = [5]
#state-space
#output [1,2,3,4] = [beta, phi, p, r]
#range aperiodic roll = index 30740 for 20 seconds


#time period of response; 1300 for spiral, 200 for rest
for i in demo:
    if i == 1 or i ==3:
        Trng = 200
    elif i == 5:
        Trng = 1300
rng = [int(meastimesdemosecs[demo]),int(meastimesdemosecs[demo])+Trng]
T = np.linspace(0,Trng/10,Trng) 

minit = 14064.8 #lbs
mused = fuelused[rng[0]]
hp0 = hp[rng[0]]
V0 = tas[rng[0]]
V = V0
m = (minit-mused)*0.45359237
rho    = rho0 * pow( ((1+(Lambda * hp0 / Temp0))), (-((g / (Lambda*R)) + 1)))  
W      = m * g    
CL = 2 * W / (rho * V0 ** 2 * S)              # Lift coefficient [ ]
CD = CD0 + (CLa * alpha0) ** 2 / (pi * A * e) # Drag coefficient [ ]
mub    = m / (rho * S * b)

mub_spiral = mub

C1 = [[(CYbdot-2*mub)*b/V, 0, 0, 0],[0, 1, 0, 0],[0, 0, -2*mub*KX2*b**2/V**2, 2*mub*KXZ*b**2/V**2],[Cnbdot*b/V, 0, 2*mub*KXZ*b**2/V**2, -2*mub*KZ2*b**2/V**2]]
C2 = [[CYb, CL, CYp*b/(2*V), (CYr-4*mub)*b/(2*V)],[0, 0, -1, 0],[Clb, 0, Clp*b/(2*V), Clr*b/(2*V)], [Cnb, 0, Cnp*b/(2*V), Cnr*b/(2*V)]]
C3 = [[CYda, CYdr], [0, 0], [Clda, Cldr],[Cnda, Cndr]]

C1inv = np.linalg.inv(C1)
A_asym2 = np.matmul(-C1inv,C2)
B_asym2 = np.matmul(-C1inv,C3)
C_asym2 = np.identity(4)
D_asym2 = np.zeros((4,2))

Aa_spiral = A_asym2

Unew = np.zeros((Trng,2))
for i in range(Trng):
    Unew[i][1] = deltar[rng[0]+i]
    Unew[i][0] = deltaa[rng[0]+i]

sys2 = cmat.ss(A_asym2, B_asym2, C_asym2, D_asym2)
response = cmat.lsim(sys2, T=T, U=-Unew, X0 = [0,rollangle[rng[0]],rollrate[rng[0]],yawrate[rng[0]]])

#plot rollangle
f, (ax1,ax2,ax3) = plt.subplots(3)
f.suptitle('Spiral motion', fontsize=16)
ax1.plot(T,np.transpose(response[0][:,1]))
ax1.plot(T,rollangle[rng[0]:rng[1]])
ax1.set_ylabel(r'rollangle ($\phi$)')
ax1.set_xlabel('Time (t)')


#plot rollrate

ax2.plot(T,np.transpose(response[0][:,2]))
ax2.plot(T,rollrate[rng[0]:rng[1]])
ax2.set_ylabel('rollrate (p)')
ax2.set_xlabel('Time (t)')


#plot yawrate

ax3.plot(T,np.transpose(response[0][:,3]))
ax3.plot(T,yawrate[rng[0]:rng[1]])
ax3.set_ylabel('yawrate (r)')
ax3.set_xlabel('Time (t)')

