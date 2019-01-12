import numpy as np

from scipy import sparse
from scipy.sparse.linalg import dsolve

import matplotlib
import matplotlib.pyplot as plt

J = 20
h = 1.0/(J+1)


###############
# Building Ah #
###############

diagonal = np.ones(J)*2.0
side_diagonal = np.ones(J-1)*(-1.0)

h2_Ah = sparse.diags([side_diagonal,diagonal,side_diagonal],
                     [-1,0,1], format="csr")

Ah = h2_Ah*(1/(h**2))


##############
# Building b #
##############

l = int(round(0.4*J))
b = np.concatenate((np.zeros(l),50*np.ones(J-2*l),np.zeros(l)));



############################
# Computing the solution u #
############################

# We solve the linear system
# See dsolve.spsolve help to understand the syntax
u = dsolve.spsolve(Ah, b)


###########################
# Plotting the solution u #
###########################

x = np.linspace(0.0, 1.0, num=J+2)
u = np.concatenate(([0], u, [0]))

fig = plt.figure()
ax = fig.gca()
ax.plot(x, u)
plt.grid(True)
plt.ylim([0,3])
plt.show()
