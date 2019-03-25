#### Analytical Eigenvalues using simplified EOM ####
from math import *
import numpy as np
from Asymmetric_Maarten import *

i = np.complex(0,1)

print("#################################")
print()

###### Aperiodic Roll motion ######
######                   ###########

print("------- Aperiodic Roll ------- ")
print()
print(" State space")
print(np.linalg.eigvals(Aa_aperiodic))
print()


mub = mub_aperiodic

Lb = Clp / (4*mub*KX2)
L = Lb*V0/b

print("Analytical")
print(L)
print()

###### Dutch Roll motion ######
######              ###########
mub = mub_dutchroll
Lknown = Lb

print("------- Dutch Roll ------- ")
print()
print(" State space")
print(np.linalg.eigvals(Aa_dutchroll))
print()


A = 8*mub**2*(KXZ**2 - KX2*KZ2)
B = 2*mub*(KXZ*Cnp + KXZ*Clr + KZ2*Clp + KX2*Cnr)
C = 1/2 * (Clr*Cnp - Cnr*Clp) - 4*mub*(KXZ*Clb + KX2*Cnb)
D = -Clb*Cnp + Clp*Cnb

f = B/A + Lknown
g = C/A + Lknown*f
h = D/A + Lknown*g # should be zero, equals 0.0131
print(h)

Lb1 = (-f + i*(-f**2 + 4*g)**0.5) / 2
Lb2 = (-f - i*(-f**2 + 4*g)**0.5) / 2

L1 = Lb1 * V0 / b
L2 = Lb2 * V0 / b

print ("Analytical")
print(L1, L2)
print()


###### Spiral motion ######
######              ###########
mub = mub_spiral

print("------- Spiral ------- ")
print()
print(" State space")
print(np.linalg.eigvals(Aa_spiral))
print()


A = 2*CL*(Clb*Cnr - Cnb*Clr)
B = CYb*(Clp*Cnr - Clr*Cnp) + (CYr - 4*mub)*(Clb*Cnp - Cnb*Clp)

Lb = A/B
L = Lb*V0/b

print ("Analytical")
print(L)




