import control.matlab as cmat
import control
import numpy as np
import matplotlib.pyplot as plt
from Cit_para import *

#define initial paramaters

V = V0

#CYb = -0.7500
#CYp = -0.0304
#CYr = 0.8495
#Clb = -0.1026
#Clp = -0.6516
#Clr = 0.2376

Clp = -0.652
Clb = -0.12
Clr = 0.12
Cnr = -0.22
Cnb = 0.16
CYb = -3.753

#define input variables for state matrices
#ybeta = V/b*CYb/2/mub
#yphi = V/b*CL/2/mub
#yp = V/b*CYp/2/mub
#yr = V/b*(CYr-4*mub)/2/mub
#yda = V/b*CYda/2/mub
#ydr = V/b*CYdr/2/mub
#
#lbeta = V/b*(Clb*KZ2 + Cnb*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
#lphi = 0
#lp = V/b*(Clp*KZ2 + Cnp*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
#lr = V/b*(Clr*KZ2 + Cnr*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
#lda = V/b*(Clda*KZ2 + Cnda*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
#ldr = V/b*(Cldr*KZ2 + Cndr*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
#
#nbeta = V/b*(Clb*KXZ + Cnb*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
#nphi = 0
#n_p = V/b*(Clp*KXZ + Cnp*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
#nr = V/b*(Clr*KXZ + Cnr*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
#nda = V/b*(Clda*KXZ + Cnda*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
#ndr = V/b*(Cldr*KXZ + Cndr*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
#
##create state-space matrix
#A_asym = [[ybeta, yphi, yp, yr], [0, 0, 2*V/b, 0], [lbeta, lphi, lp, lr], [nbeta, nphi, n_p, nr]]
#B_asym = ([[0, ydr], [0,0], [lda,ldr], [nda,ndr]])
#C_asym = np.identity(4)
#D_asym = np.zeros((4,2))
#B_asym = np.dot(B_asym, 0.025)


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
    
meastimesdemo = ['00:50:07', '00:51:13', '00:52:30', '00:55:42', '00:57:18', '00:59:43', '01:04:00']#measurement times for the demonstration part
measdemotype = ['Short period', 'Aperiodic Roll', 'Phugoid', 'Dutch Roll', 'Dutch Roll with YD', 'Spiral', 'Parabola'] #which type of manoeuvre was flown
    
def get_sec(time_str): #takes as input a timestamp string of format (H)H:MM:SS, returns equivalent number of seconds
    h, m, s = time_str.split(':') 
        
    return int(h) * 3600 + int(m) * 60 + int(s)
meastimesdemosecs = np.zeros(7)
    
for i in range(len(meastimesdemo)):
    meastimesdemosecs[i] = get_sec(meastimesdemo[i])*10    
    
    
#plt.plot(deltaa[30000:31000])
#plt.plot(rollangle[30000:31000])
    
    
    
demo = 5
#state-space  
#output [1,2,3,4] = [beta, phi, p, r]
#range aperiodic roll = index 30740 for 20 seconds
    
    
#time period of response; 1300 for spiral, 200 for rest
Trng = 1300
rng = [int(meastimesdemosecs[demo]),int(meastimesdemosecs[demo])+Trng]

T = np.linspace(0,Trng/10,Trng)
#U = np.ones((200,1))
#Uzeros = np.zeros((200,1))
Unew = np.zeros((Trng,2))
#Ua = np.concatenate((U,Uzeros),axis=1)
#Ur = np.concatenate((Uzeros,U),axis=1)
#for i in range(67,150):
#    Ur[i][1] = 0
#    Ua[i][0] = 0
for i in range(Trng):
    Unew[i][1] = deltar[rng[0]+i]
    Unew[i][0] = deltaa[rng[0]+i]
##sys = cmat.ss(A_asym, B_asym, C_asym, D_asym)
sys2 = cmat.ss(A_asym2, B_asym2, C_asym2, D_asym2)
#test1 = control.impulse_response(sys, T=T, input = 0)
#test2 = control.impulse_response(sys2, T=T, input = 0)
response = cmat.lsim(sys2, T=T, U=-Unew, X0 = [0,rollangle[rng[0]],rollrate[rng[0]],yawrate[rng[0]]])
#testroll = cmat.lsim(sys2, T=T, U=-Unew, X0 = [0,rollangle[rng[0]],rollrate[rng[0]],yawrate[rng[0]]])
    
#testresp = cmat.lsim(sys2, T=T, U=Ua, X0 = [0,0,0,0])
    
#plt.plot(T,np.transpose(test1[1][3]))
#plt.plot(T,np.transpose(test2[0][:,0]))
    
#plot response to aileron deflection (aperiodic roll)
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#
#color = 'tab:blue'
#line1 = ax1.plot(T,rollangle[rng[0]:rng[1]],label=r'$\phi$',color=color)
#ax1.set_ylabel( r' $\phi$ [$^\circ$]')
#ax1.set_xlabel('Time (t)')
#ax1.tick_params(axis='y')
#
#
#color = 'tab:orange'
#ax2 = ax1.twinx()
#ax2.set_ylabel(r'$\delta_a$ [$^\circ$]')
#line2 = ax2.plot(T,deltaa[rng[0]:rng[1]],label=r'$\delta_a$',color=color)
#ax2.tick_params(axis='y')
    #
#lns = line1 + line2
#labs = [l.get_label() for l in lns]
#ax1.legend(lns,labs)
#plt.savefig('aileroncheck.pdf')
    
    
    
    #plot response to rudder input (dutch roll)
    #demo = 3
    #rng = [int(meastimesdemosecs[demo]),int(meastimesdemosecs[demo])+Trng]
    #
    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #
    #color = 'tab:blue'
    #line1 = ax1.plot(T,yawrate[rng[0]:rng[1]],label='r',color=color)
    #ax1.set_ylabel( r'r [$^\circ/s$]')
    #ax1.set_xlabel('Time (t)')
    #ax1.tick_params(axis='y')
    #
    #
    #color = 'tab:orange'
    #ax2 = ax1.twinx()
    #ax2.set_ylabel(r'$\delta_r$ [$^\circ$]')
    #line2 = ax2.plot(T,deltar[rng[0]:rng[1]],label=r'$\delta_r$',color=color)
    #ax2.tick_params(axis='y')
    #
    #lns = line1 + line2
    #labs = [l.get_label() for l in lns]
    #ax1.legend(lns,labs)
    #plt.savefig('ruddercheck.pdf')
    
#plot rollangle
f, (ax1,ax2,ax3) = plt.subplots(3)
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
    
    
    
    #plt.plot(T,np.transpose(response[0][:,3]))
    #plt.plot(T,yawrate[rng[0]:rng[1]])
    #plt.plot(T,deltaa[rng[0]:rng[1]])
    #plt.plot(T,np.transpose(test2[0][:,2]))
    #plt.plot(T,np.transpose(test2[0][:,3]))
    
#error = []
#
#clprange = np.arange(-0.66,-0.64,0.0001)
#for a in clprange:
#    Clp = a
#    #create C1 C2 C3
#    C1 = [[(CYbdot-2*mub)*b/V, 0, 0, 0],[0, 1, 0, 0],[0, 0, -2*mub*KX2*b**2/V**2, 2*mub*KXZ*b**2/V**2],[Cnbdot*b/V, 0, 2*mub*KXZ*b**2/V**2, -2*mub*KZ2*b**2/V**2]]
#    C2 = [[CYb, CL, CYp*b/(2*V), (CYr-4*mub)*b/(2*V)],[0, 0, -1, 0],[Clb, 0, Clp*b/(2*V), Clr*b/(2*V)], [Cnb, 0, Cnp*b/(2*V), Cnr*b/(2*V)]]
#    C3 = [[CYda, CYdr], [0, 0], [Clda, Cldr],[Cnda, Cndr]]
#    
#    C1inv = np.linalg.inv(C1)
#    
#    A_asym2 = np.matmul(-C1inv,C2)
#    B_asym2 = np.matmul(-C1inv,C3)
#    C_asym2 = np.identity(4)
#    D_asym2 = np.zeros((4,2))
#    
#    #read flight data
#    time = np.loadtxt('FlightData/time.dat', dtype = 'float')
#    deltaa = np.loadtxt('FlightData/deltaa.dat', dtype = 'float')
#    deltar = np.loadtxt('FlightData/deltar.dat', dtype = 'float')
#    deltae = np.loadtxt('FlightData/deltae.dat', dtype = 'float')
#    
#    yawrate = np.loadtxt('FlightData/yawrate.dat', dtype = 'float')
#    rollangle = np.loadtxt('FlightData/rollangle.dat', dtype = 'float')
#    rollrate = np.loadtxt('FlightData/rollrate.dat', dtype = 'float')
#    pitchangle = np.loadtxt('FlightData/pitchangle.dat', dtype = 'float')
#    
#    meastimesdemo = ['00:50:07', '00:51:13', '00:52:30', '00:55:42', '00:57:18', '00:59:43', '01:04:00']#measurement times for the demonstration part
#    measdemotype = ['Short period', 'Aperiodic Roll', 'Phugoid', 'Dutch Roll', 'Dutch Roll with YD', 'Spiral', 'Parabola'] #which type of manoeuvre was flown
#    
#    def get_sec(time_str): #takes as input a timestamp string of format (H)H:MM:SS, returns equivalent number of seconds
#        h, m, s = time_str.split(':') 
#        
#        return int(h) * 3600 + int(m) * 60 + int(s)
#    meastimesdemosecs = np.zeros(7)
#    
#    for i in range(len(meastimesdemo)):
#        meastimesdemosecs[i] = get_sec(meastimesdemo[i])*10
#
#    demo = 1
#    
#    #time period of response; 1300 for spiral, 200 for rest
#    Trng = 200
#    rng = [int(meastimesdemosecs[demo]),int(meastimesdemosecs[demo])+Trng]
#    
#    T = np.linspace(0,Trng/10,Trng)
#
#    Unew = np.zeros((Trng,2))
#
#    for i in range(Trng):
#        Unew[i][1] = deltar[rng[0]+i]
#        Unew[i][0] = deltaa[rng[0]+i]
#
#    sys2 = cmat.ss(A_asym2, B_asym2, C_asym2, D_asym2)
#
#    response = cmat.lsim(sys2, T=T, U=-Unew, X0 = [0,rollangle[rng[0]],rollrate[rng[0]],yawrate[rng[0]]])
#    
#    
#    err = np.sum((np.transpose((response[0][:,3]))-yawrate[rng[0]:rng[1]])**2)+np.sum((np.transpose((response[0][:,1]))-rollangle[rng[0]:rng[1]])**2) + np.sum((np.transpose((response[0][:,2]))-rollrate[rng[0]:rng[1]])**2)
#    error.append(err)
#    print(a)
#print(clprange[np.argmin(error)])
#


clprange = np.arange(-0.66,-0.64,0.0001)
def asym_err():
    error = []
    #create C1 C2 C3
    C1 = [[(CYbdot-2*mub)*b/V, 0, 0, 0],[0, 1, 0, 0],[0, 0, -2*mub*KX2*b**2/V**2, 2*mub*KXZ*b**2/V**2],[Cnbdot*b/V, 0, 2*mub*KXZ*b**2/V**2, -2*mub*KZ2*b**2/V**2]]
    C2 = [[CYb, CL, CYp*b/(2*V), (CYr-4*mub)*b/(2*V)],[0, 0, -1, 0],[Clb, 0, Clp*b/(2*V), Clr*b/(2*V)], [Cnb, 0, Cnp*b/(2*V), Cnr*b/(2*V)]]
    C3 = [[CYda, CYdr], [0, 0], [Clda, Cldr],[Cnda, Cndr]]
    
    C1inv = np.linalg.inv(C1)
    
    A_asym2 = np.matmul(-C1inv,C2)
    B_asym2 = np.matmul(-C1inv,C3)
    C_asym2 = np.identity(4)
    D_asym2 = np.zeros((4,2))
    
    #read flight data
    time = np.loadtxt('FlightData/time.dat', dtype = 'float')
    deltaa = np.loadtxt('FlightData/deltaa.dat', dtype = 'float')
    deltar = np.loadtxt('FlightData/deltar.dat', dtype = 'float')
    deltae = np.loadtxt('FlightData/deltae.dat', dtype = 'float')
    
    yawrate = np.loadtxt('FlightData/yawrate.dat', dtype = 'float')
    rollangle = np.loadtxt('FlightData/rollangle.dat', dtype = 'float')
    rollrate = np.loadtxt('FlightData/rollrate.dat', dtype = 'float')
    pitchangle = np.loadtxt('FlightData/pitchangle.dat', dtype = 'float')
    
    meastimesdemo = ['00:50:07', '00:51:13', '00:52:30', '00:55:42', '00:57:18', '00:59:43', '01:04:00']#measurement times for the demonstration part
    measdemotype = ['Short period', 'Aperiodic Roll', 'Phugoid', 'Dutch Roll', 'Dutch Roll with YD', 'Spiral', 'Parabola'] #which type of manoeuvre was flown
    
    def get_sec(time_str): #takes as input a timestamp string of format (H)H:MM:SS, returns equivalent number of seconds
        h, m, s = time_str.split(':') 
        
        return int(h) * 3600 + int(m) * 60 + int(s)
    meastimesdemosecs = np.zeros(7)
    
    for i in range(len(meastimesdemo)):
        meastimesdemosecs[i] = get_sec(meastimesdemo[i])*10

    demo = 1
    
    #time period of response; 1300 for spiral, 200 for rest
    Trng = 200
    rng = [int(meastimesdemosecs[demo]),int(meastimesdemosecs[demo])+Trng]
    
    T = np.linspace(0,Trng/10,Trng)

    Unew = np.zeros((Trng,2))

    for i in range(Trng):
        Unew[i][1] = deltar[rng[0]+i]
        Unew[i][0] = deltaa[rng[0]+i]

    sys2 = cmat.ss(A_asym2, B_asym2, C_asym2, D_asym2)

    response = cmat.lsim(sys2, T=T, U=-Unew, X0 = [0,rollangle[rng[0]],rollrate[rng[0]],yawrate[rng[0]]])
    
    
    err = np.sum((np.transpose((response[0][:,3]))-yawrate[rng[0]:rng[1]])**2)+np.sum((np.transpose((response[0][:,1]))-rollangle[rng[0]:rng[1]])**2) + np.sum((np.transpose((response[0][:,2]))-rollrate[rng[0]:rng[1]])**2)
    error.append(err)
    
    return error

#clprange = np.arange(-0.66,-0.64,0.001)
#clp1 = []
#for i in range(len(clprange)):
#    Clp = clprange[i]
#    clp1.append(asym_err())
#Clp = clprange[np.argmin(clp1)] 
#print(Clp,'Clp')

#clbrange = np.arange(-0.2,0,0.01)
#clb1 = []
#for i in range(len(clbrange)):
#    Clb = clbrange[i]
#    clb1.append(asym_err())
#Clb = clbrange[np.argmin(clb1)] 
#print(Clb,'Clb')
#
#clrrange = np.arange(0,0.2,0.01)
#clr1 = []
#for i in range(len(clrrange)):
#    Clr = clrrange[i]
#    clr1.append(asym_err())
#Clr = clrrange[np.argmin(clr1)] 
#print(Clr,'Clr')
#
#cnrrange = np.arange(-0.3,-0.1,0.01)
#cnr1 = []
#for i in range(len(cnrrange)):
#    Cnr = cnrrange[i]
#    cnr1.append(asym_err())
#Cnr = cnrrange[np.argmin(cnr1)] 
#print(Cnr,'Cnr')
#
#cnbrange = np.arange(0.1,0.3,0.01)
#cnb1 = []
#for i in range(len(cnbrange)):
#    Cnb = cnbrange[i]
#    cnb1.append(asym_err())
#Cnb = cnbrange[np.argmin(cnb1)] 
#print(Cnb,'Cnb')

#Clp = -0.652
#Clb = -0.12
#Clr = 0.12
#Cnr = -0.22
#Cnb = 0.16
#
#cybrange = np.arange(-3.9,-3.7,0.01)
#cyb1 = []
#for i in range(len(cybrange)):
#    CYb = cybrange[i]
#    print(asym_err())
#    cyb1.append(asym_err())
#CYb = cybrange[np.argmin(cyb1)] 
#print(CYb,'CYb')
#    
    