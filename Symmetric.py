from Cit_par import *
import numpy as np
import control.matlab as cmat
import matplotlib.pyplot as plt
import sys
from Plotting import *

# STATE VECTOR
# u
# alpha
# theta
# q

# INPUT VECTOR
# delta_e

# OUTPUT VECTOR
# u
# alpha
# theta
# q

# Parameters for Matrix C1
C111 = -2 * muc 
C112 = 0
C113 = 0
C114 = 0

C121 = 0
C122 = (V0) * (CZadot - 2*muc)
C123 = 0
C124 = 0

C131 = 0
C132 = 0
C133 = -1
C134 = 0

C141 = 0
C142 = (V0) * Cmadot
C143 = 0
C144 = -2 * muc * KY2 * c

C1 = np.matrix([[C111, C112, C113, C114],
                [C121, C122, C123, C124],
                [C131, C132, C133, C134],
                [C141, C142, C143, C144]])

# Parameters for Matrix C2
C211 = (V0/c) * CXu
C212 = (V0**2/c) * CXa
C213 = (V0**2/c) * CZ0
C214 = V0 * CXq

C221 = (V0/c) * CZu
C222 = (V0**2/c) * CZa
C223 = - (V0**2/c) * CX0
C224 = V0 * (CZq + 2*muc)

C231 = 0
C232 = 0
C233 = 0
C234 = 1

C241 = (V0/c) * Cmu
C242 = (V0**2/c) * Cma
C243 = 0
C244 = (V0) * Cmq

C2 = np.matrix([[C211, C212, C213, C214],
                [C221, C222, C223, C224],
                [C231, C232, C233, C234],
                [C241, C242, C243, C244]])

# Parameters for Matrix C3
C311 = (V0**2/c) * CXde
C321 = (V0**2/c) * CZde
C331 = 0
C341 = (V0**2/c) * Cmde

C3 = np.matrix([[C311],
                [C321],
                [C331],
                [C341]])

# Matrix A
As = - np.matmul(np.linalg.inv(C1), C2)

# Matrix B
Bs = - np.matmul(np.linalg.inv(C1), C3)

# Matrix C
Cs = np.identity(4)

# Matrix D
Ds = np.zeros((4,1))

time8, data8 = givedata('Phugoid', ['aoa', 'tfulbs', 'deltae', 'pitchangle','pitchrate','tas'], False)

# Create State Space System
sys = cmat.ss(As, Bs, Cs, Ds)

T = np.arange(0, 150.1, 0.1)
X0 = [0, alpha0, th0, 0]
#U = np.ones(len(T))*-0.01
U = data8[0:1501,2]*np.pi/180

# Generate Outputs
yout, T, xout = cmat.lsim(sys, U=U, T=T, X0=X0)

u = yout[:,0]
alpha = yout[:,1]
theta = yout[:,2]
q = yout[:,3]
V = V0 + u

print(data8[0,-1])
print(data8[0,0]*np.pi/180)
print(data8[0,3]*np.pi/180)
print(data8[0,2]*np.pi/180)
# Print Eigenvalues and Plot Graphs
#plt.plot(T, V)
#plt.plot(T, data8[0:1501,-1])
#plt.xlim([T[0], T[-1]])
#plt.show()

eigs = np.linalg.eig(As)[0]
plt.scatter(eigs.real, eigs.imag)
plt.show()
