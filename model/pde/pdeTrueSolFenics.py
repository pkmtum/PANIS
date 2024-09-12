import numpy as np
import fenics as fe
import torch
import matplotlib.pyplot as plt



def solve_pde(conductivity, rhs, uBc, plotSol=False, sinX=False, options=None, idx=None):

    def func(u):
        "Return nonlinear coefficient"
        alpha = options['alpha']
        y0 = options['u0']
        return fe.exp((u-y0)*alpha)
    
    fe.set_log_level(30)
    # Define the mesh
    mesh = fe.UnitSquareMesh(conductivity.shape[0]-1, conductivity.shape[1]-1)
    #mesh = fe.UnitSquareMesh(3, 3)

    # Define the function space
    V = fe.FunctionSpace(mesh, 'P', 1)


    # Define the conductivity field as a FEniCS function
    c = fe.Function(V)

    conductivityReshaped = conductivity.reshape((-1))
    dofs = V.dofmap().dofs()
    u_coords = V.tabulate_dof_coordinates().reshape((-1, 2))
    u_nodal = np.zeros(len(dofs))

    # Find the nodal values of the FEniCS Function object that correspond to the coordinates in u_coords
    if idx == None:
        idx = []
        for i, (x, y) in enumerate(u_coords):
            idx.append(np.where((np.isclose(x, mesh.coordinates()[:, 0])) & (np.isclose(y, mesh.coordinates()[:, 1])))[0][0])

    u_nodal[:] = conductivityReshaped[idx]
        

    # Map the nodal values to the Function object
    c.vector().set_local(u_nodal)

    # Call the update() method to update the Function with the new values
    c.vector().apply('insert')


    
    if sinX == True:
        def boundaryLow(x, on_boundary):
            tol = 1E-14
            return on_boundary and fe.near(x[1], 0, tol)
        def boundaryRight(x, on_boundary):
            tol = 1E-14
            return on_boundary and fe.near(x[0], 1, tol)
        def boundaryUp(x, on_boundary):
            tol = 1E-14
            return on_boundary and fe.near(x[1], 1, tol)
        def boundaryLeft(x, on_boundary):
            tol = 1E-14
            return on_boundary and fe.near(x[0], 0, tol)

        uLow = fe.Expression('10+5*sin(pi/2*x[0])', degree=1)
        uRight = fe.Expression('10+5*sin(pi/2*x[1]+pi/2)', degree=1)
        uUp = fe.Expression('10+5*cos(pi/2*x[0]+pi)', degree=1)
        uLeft = fe.Expression('10+5*cos(pi/2*x[1]+pi/2)', degree=1)

        bcLow = fe.DirichletBC(V, uLow, boundaryLow)
        bcRight = fe.DirichletBC(V, uRight, boundaryRight)
        bcUp = fe.DirichletBC(V, uUp, boundaryUp)
        bcLeft = fe.DirichletBC(V, uLeft, boundaryLeft)
        
        bc = [bcLow, bcRight, bcUp, bcLeft]
    else:
        def boundary(x, on_boundary):
            return on_boundary
        
        bc = fe.DirichletBC(V, uBc, boundary)

    # Define the trial and test functions
    Linear = False
    if Linear:
        u = fe.TrialFunction(V)
    else:
        u = fe.Function(V)
    v = fe.TestFunction(V)


    a = c * func(u) * fe.dot(fe.grad(u), fe.grad(v))*fe.dx - (-rhs)*v*fe.dx


    L = (-rhs)*v*fe.dx
    fe.solve(a == 0, u, bc)
    

    #fe.plot(u)
    if plotSol:

        fig, axes = plt.subplots(3, 1, figsize=(14, 14))

        plt.sca(axes[0])
        fe.plot(mesh)
        numCells = int(mesh.num_cells())
        plt.title("Mesh, No cells= Num of IntPoints")

        plt.sca(axes[1])
        fe.plot(c)
        plt.title("Conductivity Field")

        plt.sca(axes[2])
        fe.plot(u)
        plt.title("Fenics Solution")
        plt.show()

        plt.savefig("./results/figs/FenicsDiscretization.png", dpi=300, bbox_inches='tight')

        plt.close('all')


    u_array = u.compute_vertex_values(mesh)
    u_grid = np.reshape(u_array, (conductivity.shape[0], conductivity.shape[1]))
    u_torch = torch.from_numpy(u_grid)

  
    return u_torch


def create_map(dimension):

  
    
    fe.set_log_level(30)
    # Define the mesh
    #mesh = fe.UnitSquareMesh(conductivity.shape[0]-1, conductivity.shape[1]-1)
    mesh = fe.UnitSquareMesh(dimension-1, dimension-1)

    # Define the function space
    V = fe.FunctionSpace(mesh, 'P', 1)

    dofs = V.dofmap().dofs()
    u_coords = V.tabulate_dof_coordinates().reshape((-1, 2))

    # Find the nodal values of the FEniCS Function object that correspond to the coordinates in u_coords
    idx = []
    for i, (x, y) in enumerate(u_coords):
        idx.append(np.where((np.isclose(x, mesh.coordinates()[:, 0])) & (np.isclose(y, mesh.coordinates()[:, 1])))[0][0])
    
    return idx