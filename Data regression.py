import numpy as np
import matplotlib.pyplot as plt
from math import *
import os
import scipy
import scipy.stats
from Cit_par import rho0,S,Gamma,Lambda,Temp0,p0,g,R,A,e

CNalpha=False
CN_CT=True
elevatortrimcurve=False

def polyfitter(x1,y1,degree):
    outlierchecks=5
    coefficients=np.flip(np.polyfit(x1,y1,degree))
    y=0
    i=0
    step=(max(x1)-min(x1))/2/len(x1)
    x=np.arange(min(x1),max(x1)+step,step)
    for cff in coefficients:
        y+=cff*x**i
        i+=1


    for j in range(outlierchecks):
        ycheck=0
        i=0
        for cff in coefficients:
            ycheck+=cff*x1**i
            i+=1
        e=y1-ycheck
        threshold=2
        deleters=np.array([])
        for i in range(len(e)):
            if abs(scipy.stats.zscore(e)[i])>threshold:
                x1=np.delete(x1,i)
                y1=np.delete(y1,i)
                print ('oulier:',i)
                break
    
    coefficients=np.flip(np.polyfit(x1,y1,degree))
    y=0
    i=0
    step=(max(x1)-min(x1))/2/len(x1)
    x=np.arange(min(x1),max(x1)+step,step)
    for cff in coefficients:
        y+=cff*x**i
        i+=1
    return x1,y1,x,y,coefficients



#1st stationairy CL-Cd
h=np.array([7000,7000,6990,6990,6900,7010]) #ft
V=np.array([251,220,190,163,143,117]) #kts
TAT=np.array([2.5,0.8,-0.8,-3.0,-3.2,-4.2]) #degree C
FFl=np.array([786.,632.,500.,454.,399.,410.]) #lbs/hr
FFr=np.array([842.,690.,552.,492.,422.,447.]) #lbs/hr
Wf=np.array([370,421,456,493,524,552]) #lbs
Winit=14064.765428 #lbs
alpha=np.array([1.4,2.1,3.3,4.9,6.6,10.3]) #degree

#stationairy elevator trim
alphae=np.array([5.3,6.3,7.3,8.5,4.5,4.1,3.4]) #degree
de=np.array([0,-0.4,0.9,-1.5,0.4,0.6,1.0]) #degree


#convert to SI
hmeters=0.3048*h
V=0.51444*V
TAT=273.15+TAT
Wf=4.44822*Wf
Winit=4.44822*Winit # newtons
FFl=0.000125998*FFl # kg/s
FFr=0.000125998*FFr # kg/s

Wmom=Winit-Wf

pmom=p0*(1+Lambda*h/Temp0)**(-g/Lambda/R)

M=(2/(Gamma-1)*((1+p0/pmom*((1+(Gamma-1)/2/Gamma*rho0/p0*V**2)**(Gamma/(Gamma-1))-1))**((Gamma-1)/Gamma)-1))**0.5
Tmom=TAT/(1+(Gamma-1)/2*M**2)
TmomISA=Temp0+Lambda*hmeters

Tdiff=Tmom-TmomISA
Vt=(Gamma*R*TmomISA)**0.5*M
rhomom=pmom/R/TmomISA
Ve=Vt*(rhomom/rho0)**0.5


CN=Wmom/(0.5*rho0*Ve**2*S)
data=np.stack((h,M,Tdiff,FFl,FFr)).T
np.savetxt('matlab.dat',data,delimiter=' ')


os.spawnl(0,"thrust(1).exe",'args')

T = np.loadtxt( 'thrust.dat' )
D=np.array([])
for line in T:
    D=np.append(D,sum(line))


CT=D/(0.5*rho0*Ve**2*S)


if CNalpha:
    degree=1
    polyfits=polyfitter(alpha,CN,degree)
    plt.plot(polyfits[0],polyfits[1],'o')
    plt.plot(polyfits[2],polyfits[3])

    print(polyfits[4])



if CN_CT:
    degree=2
    polyfits=polyfitter(CN,CT,degree)
    plt.plot(polyfits[1],polyfits[0],'o')
    plt.plot(polyfits[3],polyfits[2])

    print('lift drag polar constants: Cd0,k,1/(Pi*A*e)', polyfits[4])
    print(1/A/e/pi)

if elevatortrimcurve:
    polyfits=polyfitter(alphae,de,degree)
    degree=1
    plt.plot(polyfits[0],polyfits[1],'o')
    plt.plot(polyfits[2],polyfits[3])
    plt.gca().invert_yaxis()
    print ('-Cmalpha/Cmdelta=',polyfits[4][1])
plt.show()


