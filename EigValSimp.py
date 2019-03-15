#### Analytical Eigenvalues using simplified EOM ####
from math import *
import numpy as np
from Cit_par import *
from Symmetric import As

#print(As)

###### Short Period motion #########
print(muc)
print()
print(CZadot)
print()
print(CZq)
print()


# Eigenvalues of the state space A matrix
##print(np.linalg.eigvals(As))
##print()
      
# The EOM consists of the Z_b and M equation, with variables alpha and qc/v


# Eigenvalues of EOM matrix
A = 4 * (muc)**2 * KY2
B = - 2 * muc * (Cmq + CZa*KY2 + Cmadot)
C = CZa*Cmq - 2*muc*Cma

Lc1 = (-B + sqrt(4*A*C - B**2)) / (2*A)
Lc2 = (-B - sqrt(4*A*C - B**2)) / (2*A)

#print( Lc1, Lc2)
#print()

L1 = Lc1 * V0 / c
L2 = Lc2 * V0 / c

#print(L1,L2)

# EOM to state space and then eigenvalues of the A matrix

C111 = - 2 * muc
C121 = Cmadot
C112 = 0
C122 = - 2*muc*KY2*c/V0
C1 = np.matrix([[C111,C112],
                [C121,C122]])

C211 = CZa*V0/c
C221 = Cma*V0/c
C212 = 2*muc + CZq
C222 = Cmq
C2 = np.matrix([[C211,C212],
                [C221,C222]])

A = - (np.linalg.inv(C1)) * C2
##print(np.linalg.eigvals(A))


