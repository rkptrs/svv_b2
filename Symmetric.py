from Cit_par import *
import numpy as np
from math import *


xu=V0/c*CXu/2/muc
xalpha=V0/c*CXa/2/muc
xtheta=V0/c*CZ0/2/muc
#maybe use xq:
xq=V0/c*CXq/2/muc

zu=V0/c*CZu/(2*muc-CZadot)
zalpha=V0/c*CZa/(2*muc-CZadot)
ztheta=-V0/c*CX0/(2*muc-CZadot)
zq=V0/c*(2*muc+CZq)/(2*muc-CZadot)

mu=V0/c*(Cmu+CZu*Cmadot/(2*muc-CZadot))/(2*muc*KY2)
malpha=V0/c*(Cma+CZa*Cmadot/(2*muc-CZadot))/(2*muc*KY2)
mtheta=-V0/c*(CX0*Cmadot/(2*muc-CZadot))/(2*muc*KY2)
mq=V0/c*(Cmq+Cmadot*(2*muc+CZq)/(2*muc-CZadot))/(2*muc*KY2)





A=np.array([[xu,xalpha,xtheta,xq],
            [zu,zalpha,ztheta,zq],
            [0,0,0,V0/c],
            [mu,malpha,mtheta,mq]])



xde=V0/c*CXde/2/muc

zde=V0/c*CZde/(2*muc-CZadot)

mde=V0/c*(Cmde+CZde*Cmadot/(2*muc-CZadot))/(2*muc*KY2)


B=np.array([[xde],
            [zde],
            [0],
            [mde]])



