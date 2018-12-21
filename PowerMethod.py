import numpy as np

A = np.array([
    [ 5, 2, 0, 1],
    [-1, 3, 2, 1],
    [ 0, 0,-2, 0],
    [ 1,-1, 1, 8]])

x = np.random.rand(A.shape[1])

nb_iterations = 10

for n in range(nb_iterations):
    Ax = np.dot(A, x)
    Ax_norm = np.linalg.norm(Ax)
    x = Ax / Ax_norm;
    print ("||Ax(%d)|| = %f"%(n,Ax_norm));
