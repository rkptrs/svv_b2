import control.matlab as cmat
import control
import numpy as np
import matplotlib.pyplot as plt
from Cit_par import *



#define initial paramaters
V = 100



#define input variables for state matrices
ybeta = V/b*CYb/2/mub
yphi = V/b*CL/2/mub
yp = V/b*CYp/2/mub
yr = V/b*(CYr-4*mub)/2/mub
yda = V/b*CYda/2/mub
ydr = V/b*CYdr/2/mub

lbeta = V/b*(Clb*KZ2 + Cnb*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
lphi = 0
lp = V/b*(Clp*KZ2 + Cnp*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
lr = V/b*(Clr*KZ2 + Cnr*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
lda = V/b*(Clda*KZ2 + Cnda*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))
ldr = V/b*(Cldr*KZ2 + Cndr*KXZ)/(4*mub*(KX2*KZ2-KXZ**2))

nbeta = V/b*(Clb*KXZ + Cnb*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
nphi = 0
n_p = V/b*(Clp*KXZ + Cnp*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
nr = V/b*(Clr*KXZ + Cnr*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
nda = V/b*(Clda*KXZ + Cnda*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
ndr = V/b*(Cldr*KXZ + Cndr*KX2)/(4*mub*(KX2*KZ2-KXZ**2))

#create state-space matrix
A_asym = [[ybeta, yphi, yp, yr], [0, 0, 2*V/b, 0], [lbeta, lphi, lp, lr], [nbeta, nphi, n_p, nr]]
B_asym = [[0, ydr], [0,0], [lda,ldr], [nda,ndr]]
C_asym = np.identity(4)
D_asym = np.zeros((4,2))

#create C1 C2 C3
C1 = [[(CYbdot-2*mub)*b/V, 0, 0, 0],[0, 1, 0, 0],[0, 0, -2*mub*KX2*b**2/V**2, 2*mub*KXZ*b**2/V**2],[Cnbdot*b/V, 0, 2*mub*KXZ*b**2/V**2, -2*mub*KZ2*b**2/V**2]]
C2 = [[CYb, CL, CYp*b/(2*V), (CYr-4*mub)*b/(2*V)],[0, 0, -1, 0],[Clb, 0, Clp*b/(2*V), Clr*b/(2*V)], [Cnb, 0, Cnp*b/(2*V), Cnr*b/(2*V)]]
C3 = [[CYda, CYdr], [0, 0], [Clda, Cldr],[Cnda, Cndr]]

A_asym2 = np.matmul(-np.linalg.inv(C1),C2)
B_asym2 = np.matmul(-np.linalg.inv(C1),C3)*0.025
C_asym2 = np.identity(4)
D_asym2 = np.zeros((4,2))



#state-space  
#output [1,2,3,4] = [beta, phi, p, r]
T = np.linspace(0,15,1000)
#U = np.ones((1000,1))*(0.025)
#Uzeros = np.zeros((1000,1))
#Ua = np.concatenate((U,Uzeros),axis=1)
#Ur = np.concatenate((Uzeros,U),axis=1)
#sys = cmat.ss(A_asym, B_asym, C_asym, D_asym)
sys2 = cmat.ss(A_asym2, B_asym2, C_asym2, D_asym2)

#test1 = cmat.lsim(sys, T=T, U=Ur)
test2 = control.impulse_response(sys2, T=T, input = 0)


plt.plot(T,np.transpose(test2[1][0]))

