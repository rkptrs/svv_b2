import control
import numpy as np
import matplotlib.pyplot as plt
from Cit_par import *



#define initial paramaters
CL = 0.5
V = V0


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
np = V/b*(Clp*KXZ + Cnp*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
nr = V/b*(Clr*KXZ + Cnr*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
nda = V/b*(Clda*KXZ + Cnda*KX2)/(4*mub*(KX2*KZ2-KXZ**2))
ndr = V/b*(Cldr*KXZ + Cndr*KX2)/(4*mub*(KX2*KZ2-KXZ**2))

#create state-space matrix
A_asym = [[ybeta, yphi, yp, yr], [0, 0, 2*V/b, 0], [lbeta, lphi, lp, lr], [nbeta, nphi, np, nr]]
B_asym = [[0, ydr], [0,0], [lda,ldr], [nda,ndr]]
C_asym = np.identity(4)
D_asym = np.zeros(2,4)