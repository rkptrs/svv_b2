from math import *
import numpy as np


#####Short Period
from Symmetric_shortperiod import *

print("#################################")
print()
print("State space")
print(np.linalg.eigvals(As))
print()


###### Short Period motion ######
###### The EOM consists of the Z_b and M equation, with variables alpha and qc/v ###########

A = 4 * (muc)**2 * KY2
B = - 2 * muc * (Cmq + CZa*KY2 + Cmadot) - CZq*Cmadot
C = CZa*Cmq - (2*muc + CZq)*Cma

Lc1 = (-B + (B**2 - 4*A*C)**0.5) / (2*A)
Lc2 = (-B - (B**2 - 4*A*C)**0.5) / (2*A)

L1 = Lc1 * V0 / c
L2 = Lc2 * V0 / c

print("Analytical")
print(L1,L2)
print()





###### Phugoid Motion ######
from Symmetric_phugoid import *
###### The EOM consist of X, Z and kinematic euquation with variables U, theta and q.


print("State space")
print(np.linalg.eigvals(As))
print()

A = 2*muc*(CZa*Cmq - (CZq + 2*muc)*Cma)
B = 2*muc*(CXu*Cma - CXa*Cmu) - CXu*(CZa*Cmq - CZq*Cma) + CXa*(CZu*Cmq - CZq*Cmu) - CXq*(CZu*Cma - CZa*Cmu)

C = CZ0*(Cmu*CZa-CZu*Cma)

Lc1 = (-B + (B**2 - 4*A*C)**0.5) / (2*A)
Lc2 = (-B - (B**2 - 4*A*C)**0.5) / (2*A)

L1 = Lc1 * V0 / c
L2 = Lc2 * V0 / c

print ("Analytical")
print(L1,L2)
print()







