import numpy as np
from scipy import sparse
from scipy.sparse.linalg import dsolve

import matplotlib
import matplotlib.pyplot as plt

h_tab = []   # Where we will store the step sizes
err_tab = [] # Where we will store the errors (infinity norm)

for J in [10, 15, 20, 30, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 50000, 100000]:

    h = 1.0/(J+1)

    # Building Ah
    diagonal = np.ones(J)*2.0
    side_diagonal = np.ones(J-1)*(-1.0)
    h2_Ah = sparse.diags([side_diagonal,diagonal,side_diagonal],
                         [-1,0,1], format="csr")
    Ah = h2_Ah*(1/(h**2))

    # Building b
    b = np.ones(J)

    # Solving for u
    u = dsolve.spsolve(Ah, b)

    # We know the solution explicitely so we compute it
    x = np.linspace(1/(J+2), 1-1/(J+2), num=J)
    solution = (-x**2+x)/2

    # We compute the error and its infinity norm
    error = abs(u-solution).max()

    # Printing the results and updating our arrays
    print("h = %e   log(h) = %e   Error = %e   log(Error) = %e" %(h,np.log10(h),error,np.log10(error)))
    h_tab.append(h)
    err_tab.append(error)

# We plot the error vs. h
fig = plt.figure()
ax = fig.gca()
plt.xlabel('h')
plt.ylabel('||E||_infini')
ax.plot(h_tab, err_tab, '.')
plt.show()

# We look at the log-scale plot
h_tab_log = np.log10(h_tab)
err_tab_log = np.log10(err_tab)

fig = plt.figure()
ax = fig.gca()
plt.xlabel('log(h)')
plt.ylabel('log(||E||_infini)')
ax.plot(h_tab_log, err_tab_log, '.')
plt.show()

# Trying to estimate the slope of what looks like a line in the log-scale plot
# That should be close to 2 according to what we did in class
print("Order = ",(err_tab_log[len(err_tab_log)-1]-err_tab_log[0])/(h_tab_log[len(h_tab_log)-1]-h_tab_log[0]))
