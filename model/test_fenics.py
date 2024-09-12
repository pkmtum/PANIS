"""
Manthos and Vincent are having fun with FEniCS!!1!!1111 Vol #1

"""
import warnings

"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary
  u_D = 1 + x^2 + 2y^2
    f = -6
"""

import fenics as df
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import time


# %% General setup
# Create mesh and define function space
mesh = df.IntervalMesh(5, 0.0, 1.0)
# displacement FunctionSpace
V = df.FunctionSpace(mesh, "CG", 1)
# material function space
Vc = df.FunctionSpace(mesh, "DG", 0)

# %% Boundary conditions
# Define boundary condition
u_D_left = df.Expression('1', degree=0)  # x is domain coordinates
u_D_right = df.Expression('1', degree=0)  # x is domain coordinates

tol = 1e-6

def boundary_left(x, on_boundary):
    return on_boundary and (df.near(x[0], 0, tol))


def boundary_right(x, on_boundary):
    return on_boundary and (df.near(x[0], 1, tol))


bc1 = df.DirichletBC(V, u_D_left, boundary_left)
bc2 = df.DirichletBC(V, u_D_right, boundary_right)
bc = [bc1, bc2]

# %% FE stuff
# Define variational problem
w = df.TrialFunction(V)
y = df.TestFunction(V)

# a random constant
f = df.Constant(1.0)

# weak formulation
a = 2*df.dot(df.grad(w), df.grad(y)) * df.dx
L = f * y * df.dx

# Compute solution
y = df.Function(V)

# I want it to give it my custom vector
y_numpy = np.random.rand(11)
# y_fun = df.Function(V)
# y_fun.vector().set_local(y_numpy)
t1 = time.time()
A, b = df.assemble_system(a, L, bc)
print(time.time()-t1)
A_numpy = A.array()
b_numpy = b.get_local()
print(A_numpy)
print(b_numpy)

aa = np.array([[5],[0],[0],[5]])
f = -1*np.array([[0.2],[0.2],[0.2],[0.2]])
b =  2. * aa - f
print(b)
# res = np.dot(A_numpy, y_numpy) - b_numpy

df.solve(a == L, y, bc)

# the TRUE residual
u_numpy = y.vector().get_local()
t2 = time.time()
res = np.dot(A_numpy, u_numpy) - b_numpy
print(time.time()-t2)
print(res)

# %% Plotting
# Plot solution and mesh
df.plot(mesh)
# Hold plot
plt.show()


# solution in numpy
dofs = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
plt.plot(dofs, u_numpy)
plt.grid()
plt.show()
