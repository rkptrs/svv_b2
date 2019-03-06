from Cit_par import *
import numpy as np
import control.matlab as cmat
import matplotlib.pyplot as plt

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
C111 = -2 * muc * (c/V0**2)
C112 = 0
C113 = 0
C114 = 0

C121 = 0
C122 = (c/V0) * (CZadot - 2*muc)
C123 = 0
C124 = 0

C131 = 0
C132 = 0
C133 = -(c/V0)
C134 = 0

C141 = 0
C142 = (c/V0) * Cmadot
C143 = 0
C144 = -2 * (c/V0)**2 * muc * KY2

C1 = np.matrix([[C111, C112, C113, C114],
                [C121, C122, C123, C124],
                [C131, C132, C133, C134],
                [C141, C142, C143, C144]])

# Parameters for Matrix C2
C211 = (1/V0) * CXu
C212 = CXa
C213 = CZ0
C214 = (c/V0) * CXq

C221 = (1/V0) * CZu
C222 = CZa
C223 = -CX0
C224 = (c/V0) * (CZq + 2*muc)

C231 = 0
C232 = 0
C233 = 0
C234 = (c/V0)

C241 = (1/V0) * Cmu
C242 = Cma
C243 = 0
C244 = (c/V0) * Cmq

C2 = np.matrix([[C211, C212, C213, C214],
                [C221, C222, C223, C224],
                [C231, C232, C233, C234],
                [C241, C242, C243, C244]])

# Parameters for Matrix C3
C311 = CXde
C321 = CZde
C331 = 0
C341 = Cmde

C3 = np.matrix([[C311],
                [C321],
                [C331],
                [C341]])

# Matrix A
A = - np.matmul(np.linalg.inv(C1), C2)

# Matrix B
B = - np.matmul(np.linalg.inv(C1), C3)

# Matrix C
C = np.identity(4)

# Matrix D
D = np.zeros((4,1))

# Create State Space System
sys = cmat.ss(A, B, C, D)

T = np.linspace(0, 150, 1000)
X0 = [V0*np.cos(alpha0), alpha0, th0, 0]
U = np.hstack((np.ones(50) * -0.005, np.zeros(950)))

# Make Output Plots
yout, T, xout = cmat.lsim(sys, U=U, T=T, X0=X0)
yout0 = yout[:,0]
yout1 = yout[:,1]
yout2 = yout[:,2]
yout3 = yout[:,3]

V = (yout0 / np.cos(yout3))

print(np.linalg.eig(A)[0])
#plt.scatter(np.real(np.linalg.eig(A)[0]), np.imag(np.linalg.eig(A)[0]))
plt.plot(T, V)
plt.show()

