import numpy as np
import matplotlib.pyplot as plt
from Cit_par import rho0,S,gamma,Lambda,Temp0,p0,g,R

##k=20
##x=np.arange(0,1.5,1/k)
##degree=3 # between 1 and 5
##x1 = x
##y1 = x1 - 2 * (x1 ** 2) + 0.5 * (x1 ** 3)



def polyfitter(x1,y1,degree):
    coefficients=np.flip(np.polyfit(x1,y1,degree))
    y=0
    i=0
    step=(max(x1)-min(x1))/2/len(x1)
    x=np.arange(min(x1)-3,max(x1)+step,step)
    for cff in coefficients:
        y+=cff*x**i
        i+=1
    return x,y,coefficients




h=np.array([7000,7000,6990,6990,6900,7010]) #ft
V=np.array([251,220,190,163,143,117]) #kts
TAT=np.array([2.5,0.8,-0.8,-3.0,-3.2,-4.2]) #degree C
FFl=np.array([786.,632.,500.,454.,399.,410.]) #lbs/hr
FFr=np.array([842.,690.,552.,492.,422.,447.]) #lbs/hr
Wf=np.array([370,421,456,493,524,552]) #lbs
Winit=14064.765428 #lbs
alpha=np.array([1.4,2.1,3.3,4.9,6.6,10.3]) #degree


#convert to SI
hmeters=0.3048*h
V=0.51444*V
TAT=273.15+TAT
Wf=4.44822*Wf
Winit=4.44822*Winit # newtons


Wmom=Winit-Wf

pmom=p0*(1+Lambda*h/Temp0)**(-g/Lambda/R)

M=(2/(gamma-1)*((1+p0/pmom*((1+(gamma-1)/2/gamma*rho0/p0*V**2)**(gamma/(gamma-1))-1))**((gamma-1)/gamma)-1))**0.5
Tmom=TAT/(1+(gamma-1)/2*M**2)
TmomISA=Temp0+Lambda*hmeters

Tdiff=Tmom-TmomISA

CN=Wmom/(0.5*rho0*V**2*S)

degree=1

plt.plot(alpha,CN,'o')
plt.plot(polyfitter(alpha,CN,degree)[0],polyfitter(alpha,CN,degree)[1])
#plt.show()

#print(polyfitter(alpha,CN,degree)[2])

data=np.stack((h,M,Tdiff,FFl,FFr)).T
np.savetxt('matlab.dat',data,delimiter=' ')
