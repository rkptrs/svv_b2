#### Analytical Eigenvalues using simplified EOM ####
from math import *
import numpy as np
from Asymmetric_Maarten import *

print("#################################")
print()
### Eigenvalues of the state space A matrix
print("State space eigenvalues: ", np.linalg.eigvals(Aa))
print()
print("#################################")
print()


###### Aperiodic Roll motion ######
######                   ###########


Lb = Clp / (4*mub*KX2)
L = Lb*V0/b

print("Analytical Aperiodic Roll eigenvalues: ",L)
print()
print("#################################")
print()

###### Dutch Roll motion ######
######              ###########
Lknown = Lb


A = 8*mub**2*(KXZ**2 - KX2*KZ2)
B = 2*mub*(KXZ*Cnp + KXZ*Clr + KZ2*Clp + KX2*Cnr)
C = 1/2 * (Clr*Cnp - Cnr*Clp) - 4*mub*(KXZ*Clb + KX2*Cnb)
D = -Clb*Cnp + Clp*Cnb

f = B/A + Lknown
g = C/A + Lknown*f
h = D/A + Lknown*g # should be zero, equals 0.005009

Lb1 = (-f + (f**2 - 4*g)**0.5) / 2
Lb2 = (-f - (f**2 - 4*g)**0.5) / 2

L1 = Lb1 * V0 / b
L2 = Lb2 * V0 / b

print ("Analytical Dutch Roll eigenvalues: ",L1,L2)
print()
print("#################################")


###### Spiral motion ######
######              ###########

A = 2*CL*(Clb*Cnr - Cnb*Clr)
B = CYb*(Clp*Cnr - Clr*Cnp) + (CYr - 4*mub)*(Clb*Cnp - Cnb*Clp)

Lb = A/B
L = Lb*V0/b

print ("Analytical Spiral motion eigenvalues: ",L)
print()
print("#################################")


