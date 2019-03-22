#### Analytical Eigenvalues using simplified EOM ####
from math import *
import numpy as np
from Symmetric_Maarten import *
print("#################################")
print()
# Eigenvalues of the state space A matrix
print("State space eigenvalues: ", np.linalg.eigvals(As))
print()
print("#################################")
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

print("Analytical Short period eigenvalues: ",L1,L2)
print()
print("#################################")
print()
###### Phugoid Motion ######
###### The EOM consist of X, Z and kinematic euquation with variables U, theta and q.

##A = -2*muc*(CZq + 2*muc)
##B = CXu*(CZq + 2*muc) - CXq*CZu
##C = -CZ0*CXu

A = 2*muc*(CZa*Cmq - 2*muc*Cma)
B = 2*muc*(CXu*Cma - Cmu*CXa) + Cmq*(CZu*CXa - CXu*CZa)
C = CZ0*(Cmu*CZa-CZu*Cma)

Lc1 = (-B + (B**2 - 4*A*C)**0.5) / (2*A)
Lc2 = (-B - (B**2 - 4*A*C)**0.5) / (2*A)

L1 = Lc1 * V0 / c
L2 = Lc2 * V0 / c

print ("Analytical Phugoid eigenvalues: ",L1,L2)
print()
print("#################################")






