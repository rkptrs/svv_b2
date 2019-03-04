from Cit_par import *
import numpy as np
from math import *
import control.matlab as cmat
import matplotlib.pyplot as plt

# STATE VECTOR
# uhat
# alpha
# theta
# qcbar/V

# INPUT VECTOR
# delta_e

# Parameters for Matrix A
xu = (V0/c) * (CXu / (2*muc))
xalpha = (V0/c) * (CXa / (2*muc))
xtheta= (V0/c) * (CZ0 / (2*muc))
xq= (V0/c) * (CXq / (2*muc))
zu= (V0/c) * (CZu / (2*muc-CZadot))
zalpha = (V0/c) * (CZa / (2*muc-CZadot))
ztheta = -(V0/c) * (CX0 / (2*muc-CZadot))
zq = (V0/c) * ((2*muc+CZq) / (2*muc-CZadot))
mu = (V0/c) * ((Cmu + CZu * (Cmadot/(2*muc-CZadot))) / (2*muc*KY2))
malpha = (V0/c) * ((Cma + CZa * (Cmadot/(2*muc-CZadot))) / (2*muc*KY2))
mtheta = -(V0/c) * ((CX0 * (Cmadot/(2*muc-CZadot))) / (2*muc*KY2))
mq= (V0/c) * ((Cmq + Cmadot * ((2*muc+CZq)/(2*muc-CZadot))) / (2*muc*KY2))

# Parameters for Matrix B
xde = (V0/c) * (CXde / (2*muc))
zde = (V0/c) * (CZde / (2*muc-CZadot))
mde = (V0/c) * ((Cmde + CZde * (Cmadot/(2*muc-CZadot))) / (2*muc*KY2))

# Matrix A
A = np.array([[xu, xalpha, xtheta, xq],
              [zu, zalpha, ztheta, zq],
              [0, 0, 0, V0/c],
              [mu, malpha, mtheta, mq]])

# Matrix B
B = np.array([[xde],
              [zde],
              [0],
              [mde]])

# Matrix C
C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

#Matrix D
D = np.array([[0],
              [0],
              [0],
              [0]])


#Make State Space System
T = np.linspace(0, 100, 1000)
U = np.ones(1000)*(np.pi/180)
X0 = [np.cos(alpha0), alpha0, th0, 0]

sys = cmat.ss(A, B, C, D)

#Make Output Vectors
test = cmat.lsim(sys, T=T, U=U, X0=X0)
out0 = [val[0] for val in test[0]]
out1 = [val[1] for val in test[0]]
out2 = [val[2] for val in test[0]]
out3 = [val[3] for val in test[0]]

plt.plot(T, out0)
plt.show()
