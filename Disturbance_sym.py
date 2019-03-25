from Cit_par import *
import numpy as np
import control.matlab as cmat
import matplotlib.pyplot as plt

##################
# PHUGOID MOTION #
##################
# STATE VECTOR   #
# u              #
# alpha          #
# theta          #
# q              #
#                #
# INPUT VECTOR   #
# delta_e        #
#                #
# OUTPUT VECTOR  #
# u              #
# alpha          #
# theta          #
# q              #
##################

# Tweaking To Make The Model Fit
CZu = -0.59
CXa = -0.8
CXu = -0.05
Cma = -0.55


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

# Create State Space System
sys = cmat.ss(As, Bs, Cs, Ds)

# Generate Outputs
T = np.arange(0, 150.1, 0.1)
X0 = [0, alpha0, th0, 0]
U = np.ones(len(T)) * -0.005
#V0 = 91.271995
#W0 = 5976.97

yout, T, xout = cmat.lsim(sys, U=U, T=T, X0=X0)

u = yout[:,0]
alpha = yout[:,1] * 180/np.pi
theta = yout[:,2] * 180/np.pi
q = yout[:,3] * 180/np.pi
V = V0 + u

modeldata = np.zeros((1501, 6))
modeldata[:,0] = T
modeldata[:,1] = V
modeldata[:,2] = alpha
modeldata[:,3] = theta
modeldata[:,4] = q
modeldata[:,5] = U

plt.subplot(511) #V
plt.plot(modeldata[:,0], modeldata[:,1], color='black') #model data
plt.xlabel('Time [s]')
plt.ylabel('V [m/s]')
plt.xlim([modeldata[0,0], modeldata[-1,0]])
plt.grid()
   
plt.subplot(512) #alpha
plt.plot(modeldata[:,0], modeldata[:,2], color='black') #model data
plt.xlabel('Time [s]')
plt.ylabel('α [°]')
plt.xlim([modeldata[0,0], modeldata[-1,0]])
plt.grid()
    
plt.subplot(513) #theta
plt.plot(modeldata[:,0], modeldata[:,3], color='black') #model data
plt.xlabel('Time [s]')
plt.ylabel('θ [°]')
plt.xlim([modeldata[0,0], modeldata[-1,0]])
plt.grid()
    
plt.subplot(514) #q
plt.plot(modeldata[:,0], modeldata[:,4], color='black') #model data
plt.xlabel('Time [s]')
plt.ylabel('q [°/s]')
plt.xlim([modeldata[0,0], modeldata[-1,0]])
plt.grid()
    
plt.subplot(515) #deltae
plt.plot(modeldata[:,0], modeldata[:,5], color='black') #model data
plt.xlabel('Time [s]')
plt.ylabel('δe [°]')
plt.xlim([modeldata[0,0], modeldata[-1,0]])
plt.grid()
    
plt.show()


