import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# speed coefficient
alpha = 1.0


# Spatial domain
x_min = 0.0
x_max = 1.0

Jx = 20
hx = (x_max-x_min)/(Jx+1)


# Time domain
t_min = 0.0
t_max = 3.0

Jt = 3000
ht = (t_max-t_min)/(Jt+1)

print("alpha * ht / hx^2 =", alpha*ht/(hx**2)) 


# Each value of the U array contains the solution for spatial values at each timestep
# U[0][j] contains the discretization of the initial condition (here 0)

U = []
U.append(np.zeros(Jx))



# Discretizing the heat source term
# 0  on [0.0, 0.4]
# 50 on ]0.4, 0.6[
# 0  on [0.6, 1.0]

l = int(round(0.4*Jx))
F = np.concatenate((np.zeros(l),50*np.ones(Jx-2*l),np.zeros(l)));


# Computing U

for i in range(Jt):
    
    Ui = [] # u values for this time step
    
    for j in range(Jx):

        if j == 0: # left boundary
            Ui.append(U[i-1][j] + (alpha*ht/(hx**2))*(-2*U[i-1][j]+U[i-1][j+1]) + ht*F[j])
        elif j == Jx-1: # right boundary
            Ui.append(U[i-1][j] + (alpha*ht/(hx**2))*(U[i-1][j-1]-2*U[i-1][j]) + ht*F[j])
        else:
            Ui.append(U[i-1][j] + (alpha*ht/(hx**2))*(U[i-1][j-1]-2*U[i-1][j]+U[i-1][j+1]) + ht*F[j])

    U.append(Ui)



# Plot solution
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

# Animate the data
k = 0
def animate(i):
    global k
    x = np.concatenate(([0], U[k], [0]))
    k += 1
    ax1.clear()
    plt.plot(np.linspace(x_min,x_max,Jx+2),x,color='red')
    plt.grid(True)
    plt.ylim([0,3])
    plt.xlim([x_min,x_max])
    plt.legend(["i=%d"%i])
    
anim = animation.FuncAnimation(fig,animate,frames=Jt-1)
plt.show()
