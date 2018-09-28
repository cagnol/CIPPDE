# -*- coding: utf-8 -*-
"""
The Euler Forward Method
CentraleSupelec 2018-2019
John Cagnol and Corentin Jeudy
"""


import math
import numpy as np
import matplotlib.pyplot as plt


################################################################################
# DEFINE THE ODE                                                               #
################################################################################

# y' = f(t,y)
# y(t0) = t0

# Define the function f
def f(time,x):
    """
    - brief: Computes the value for the ODE defined by y' = -3y 
    - input: (float) time, (float) x
    - output: (float) -3 * x
    """
    return -3 * x

# Define t0 and y0
time_origin = 0
initial_condition = 1


################################################################################
# DEFINE THE MESH                                                              #
################################################################################

simulation_time = 100    # End of the interval [t0,T]
number_iterations = 100    # Number of iterations
time_step = simulation_time/number_iterations  

# Define the arrays holding the values of t and y at each point of the mesh
time_array = np.empty(number_iterations)
numerical_solution = np.empty(number_iterations)

print("Time origin t0 = ",time_origin)
print("simulation time T = ",simulation_time)
print("Time step dt = ",time_step)


################################################################################
# APPROXIMATE THE SOLUTION USING THE EULER FORWARD METHOD                      #
################################################################################

numerical_solution[time_origin] = initial_condition

for i in range(0, number_iterations - 1):
    time_array[i+1] = time_array[i] + time_step
    numerical_solution[i+1] = numerical_solution[i] + time_step*f(time_array[i],numerical_solution[i])


################################################################################
# COMPARE THE APPROXIMATE SOLUTION TO THE EXACT SOLUTION                       #
################################################################################

# Array to hold the values of the exact solution at each point of the mesh
exact_solution = np.zeros([number_iterations])

# Array to hold the difference between exact and approximate solution (error)
error_array = np.zeros([number_iterations])

# Compute the exact solution at each point of the mesh, and the error.
for i in range(0, number_iterations - 1):
    exact_solution[i] = initial_condition * math.exp( -3 * time_array[i])
    error_array[i] = abs(numerical_solution[i] - exact_solution[i])

print("Error max: {}".format(error_array.max()))


################################################################################
# PLOT THE APPROXIMATE SOLUTION AND THE EXACT SOLUTION                         #
################################################################################

figure = plt.figure()

plt.title("Comparison of solutions for the ODE $y' = -3y$")

plt.grid(True)
plt.xlabel("Time $t$")

plt.plot(time_array,numerical_solution,'b', label = "Numerical solution $z$")
plt.plot(time_array,exact_solution,'r', label = "Exact solution $y$")
plt.legend(loc = "upper left")

plt.show()
