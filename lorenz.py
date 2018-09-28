# -*- coding: utf-8 -*-
"""
Lorenz
CentraleSupelec 2018-2019
John Cagnol and Corentin Jeudy
"""

"""
IN THE CODE:
- CR stands for Convection Rate
- HTV stands for Horizontal Temperature Variation
- VTV stands for Vertical Temperature Variation

These abbreviations are solely used to make the improve the code readability.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define the parameters
sigma = 10.0
rho   = 28.0
beta  = 8.0/3.0

# Define t0 and the initial conditions
time_origin = 0
initial_CR = 1.0
initial_HTV = 1.0
initial_VTV = 1.0

# Define de Mesh and print its characteristics 
simulation_time = 50    # End of the interval [t0,T]
number_iterations = 5000    # Number of iterations
time_step = simulation_time/number_iterations   
print("Time origin t0 = ",time_origin)
print("simulation time T = ",simulation_time)
print("Time step dt = ",time_step)

# Define the arrays holding the values of t and y at each point of the mesh
time_array = np.empty(number_iterations)
CR = np.empty(number_iterations)
HTV = np.empty(number_iterations)
VTV = np.empty(number_iterations)

# Approximate the solution using the Euler Forward Method
time_array[0] = time_origin
CR[0] = initial_CR
HTV[0] = initial_HTV
VTV[0] = initial_VTV

for i in range(0, number_iterations - 1):
    time_array[i+1] = time_array[i] + time_step
    CR[i+1] = CR[i] + time_step*sigma*(HTV[i] - CR[i])
    HTV[i+1] = HTV[i] + time_step*(CR[i]*(rho - VTV[i]) - HTV[i])
    VTV[i+1] = VTV[i] + time_step*(CR[i]*HTV[i] - beta*VTV[i])

# Plot the (approximate) solution
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("Convection rate $x$")
ax.set_ylabel("Horizontal temperature variation $y$")
ax.set_zlabel("Vertical temperature variation $z$")
ax.plot(CR, HTV, VTV)
plt.show()
