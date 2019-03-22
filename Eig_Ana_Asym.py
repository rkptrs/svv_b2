#### Analytical Eigenvalues using simplified EOM ####
from math import *
import numpy as np
from Asymmetric_Maarten import *

print("#################################")
print()
# Eigenvalues of the state space A matrix
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



##Lb1 = (-B + (B**2 - 4*A*C)**0.5) / (2*A)
##Lb2 = (-B - (B**2 - 4*A*C)**0.5) / (2*A)
##
##L1 = Lc1 * V0 / b
##L2 = Lc2 * V0 / b

##print ("Analytical Dutch Roll eigenvalues: ",L1,L2)
##print()
##print("#################################")


###### Spiral motion ######
######              ###########

A = 2*CL*(Clb*Cnr - Cnb*Clr)
B = CYb*(Clp*Cnr - Clr*Cnp) + (CYr - 4*mub)*(Clb*Cnp - Cnb*Clp)

Lb = A/B
L = Lb*V0/b

print ("Analytical Spiral motion eigenvalues: ",L)
print()
print("#################################")


