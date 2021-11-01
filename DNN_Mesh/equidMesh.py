from numpy.lib.nanfunctions import _nanquantile_ureduce_func
from UtilityFunctions import *
import os
import timeit
import numpy as np


def equid(Ns, interp_bound=1):

    print(f"Running with interp_bound {interp_bound}")
    hmin = np.zeros(len(Ns))
    L2err = np.zeros(len(Ns))
    dim = np.zeros(len(Ns))
    time = np.zeros(len(Ns))

    abstol = 1e-16
    max_iter = 50

    for it, N in enumerate(Ns):

        mesh = UnitIntervalMesh(N)

        DG0 = FunctionSpace(mesh, 'DG', 0)
        CG1 = FunctionSpace(mesh, 'CG', 1)
        CG5 = FunctionSpace(mesh, 'CG', 5)

        #u_expr = Expression_uexact(exponent= 2.0/3.0 ,rho= 1.0,degree=5)
        u_expr = Expression('pow(x[0],2.0/3.0)', degree=5)
        u_ex = Function(CG5)

        u_ex.interpolate(u_expr)

        dof2vertex_map = dof_to_vertex_map(CG1)

        # Solve 1D Poisson equation
        f = -div(grad(u_ex))
        u = solve_poisson(CG1, mesh, u_expr, f)

        iteration = 0
        tol = 1.0
        t0 = timeit.default_timer()

        while (tol > abstol) and (iteration < max_iter):

            print('rel_tol is: ', tol)

            xi = mesh.coordinates()[:, 0]
            if interp_bound:
                w = monitor_interpolant(mesh, DG0, CG1, u_ex)
            else:
                w = monitor_posteriori(mesh, DG0, CG1, u, f)

            #w_post = smoothing(w_post,CG1)

            x = equidistribute(xi, w, dof2vertex_map)

            if iteration > 0:
                tol = max(abs(x - x_prev))

            mesh.coordinates()[:, 0] = x

            u_ex.interpolate(u_expr)
            f = -div(grad(u_ex))
            u = solve_poisson(CG1, mesh, u_expr, f)

            x_prev = x
            iteration += 1

        time[it] = timeit.default_timer() - t0
        # compute errors in L2 norm
        dx = Measure('dx', domain=mesh)
        L2err[it] = np.sqrt(assemble((u_expr*u_expr + u*u - 2*u_expr*u)
                                     * dx(mesh), form_compiler_parameters={"quadrature_degree": 5}))
        hmin[it] = min(abs(np.diff(np.sort(x))))
        dim[it] = CG1.dim()

    return dim, L2err, hmin, time


def refinement(mesh, DG0, u, f, ref_ratio=False, tol=1.0):

    dx = Measure('dx', domain=mesh)

    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    w = TestFunction(DG0)

    cell_residual = Function(DG0)

    # assume avg diameter close to edge length
    residual = h**2*w*(u.dx(0).dx(0)+f)**2*dx
    assemble(residual, tensor=cell_residual.vector())

    theta = sum(cell_residual.vector()[:])/mesh.num_cells()

    # Mark cells for refinement
    cell_markers = MeshFunction('bool', mesh, mesh.topology().dim())

    if ref_ratio:

        gamma_0 = sorted(cell_residual.vector()[:], reverse=True)[
            int(mesh.num_cells()*ref_ratio)]
        gamma_0 = MPI.max(mesh.mpi_comm(), gamma_0)

        for c in cells(mesh):
            cell_markers[c.index()] = cell_residual.vector()[
                :][c.index()] > gamma_0

    else:
        # Apply equidistribution
        for c in cells(mesh):
            cell_markers[c.index()] = cell_residual.vector()[
                :][c.index()] > theta*tol

    # Refine mesh
    mesh = refine(mesh, cell_markers)

    return mesh


def h_refinement(N, n_ref, ref_ratio=False, tol=1.0):

    hmin = np.zeros(n_ref)
    L2err = np.zeros(n_ref)
    dim = np.zeros(n_ref)
    time = np.zeros(n_ref)
    t0 = timeit.default_timer()
    mesh = UnitIntervalMesh(N)  # mesh a-posteriori error

    t0 = timeit.default_timer()
    u_expr = Expression('pow(x[0],2.0/3.0)', degree=5)

    for it in range(n_ref):

        DG0 = FunctionSpace(mesh, 'DG', 0)
        CG1 = FunctionSpace(mesh, 'CG', 1)
        CG5 = FunctionSpace(mesh, 'CG', 5)
        u_ex = Function(CG5)

        # Solve 1D Poisson equation
        u_ex.interpolate(u_expr)
        f = -div(grad(u_ex))
        u_ref = solve_poisson(CG1, mesh, u_expr, f)

        dx = Measure('dx', domain=mesh)
        L2err[it] = np.sqrt(assemble((u_expr*u_expr + u_ref*u_ref - 2*u_expr*u_ref)
                                     * dx(mesh), form_compiler_parameters={"quadrature_degree": 5}))
        dim[it] = CG1.dim()

        mesh = refinement(mesh, DG0, u_ref, f, ref_ratio=0.25)
        time[it] = timeit.default_timer() - t0
        x = mesh.coordinates()[:, 0]
        dof2vertex_map = dof_to_vertex_map(CG1)
        hmin[it] = min(np.diff(np.sort(x)))

    return dim, L2err, hmin, time


def uniform_refinement(N, n_ref):

    hmin = np.zeros(n_ref)
    L2err = np.zeros(n_ref)
    dim = np.zeros(n_ref)
    time = np.zeros(n_ref)
    t0 = timeit.default_timer()
    mesh = UnitIntervalMesh(N)

    for it in range(n_ref):
        DG0 = FunctionSpace(mesh, 'DG', 0)
        CG1 = FunctionSpace(mesh, 'CG', 1)
        CG5 = FunctionSpace(mesh, 'CG', 5)
        u_ex = Function(CG5)
        u_expr = Expression('pow(x[0],2.0/3.0)', degree=5)
        # Solve 1D Poisson equation
        u_ex.interpolate(u_expr)
        f = -div(grad(u_ex))
        u_ref = solve_poisson(CG1, mesh, u_expr, f)

        dx = Measure('dx', domain=mesh)
        L2err[it] = np.sqrt(assemble((u_expr*u_expr + u_ref*u_ref - 2*u_expr*u_ref)
                                     * dx(mesh), form_compiler_parameters={"quadrature_degree": 5}))
        dim[it] = CG1.dim()

        mesh = refine(mesh)
        time[it] = timeit.default_timer() - t0

        x = mesh.coordinates()[:, 0]
        dof2vertex_map = dof_to_vertex_map(CG1)
        hmin[it] = min(np.diff(np.sort(x)))

    return dim, L2err, hmin, time


Ns = 2**(np.arange(4, 13))

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

folder_names = ['/post', '/interp']

for i in range(2):
    dim, L2err, hmin, time = equid(Ns, interp_bound=i)
    folder_name = folder_names[i]
    print(f"print in {folder_name}")

    path_folder = path + folder_name
    if not(os.path.exists(path_folder)):
        os.makedirs(path_folder)

    np.savez(path_folder + f"{folder_name}.npz",
             dim=dim, L2err=L2err, hmin=hmin, time=time)


N = 2**4
n_ref = 25

dim, L2err, hmin, time = h_refinement(N, n_ref, ref_ratio=False, tol=1.0)


folder_name = '/href'
path_folder = path + folder_name

if not(os.path.exists(path_folder)):
    os.makedirs(path_folder)
np.savez(path_folder + f"{folder_name}.npz",
         dim=dim, L2err=L2err, hmin=hmin, time=time)

N = 2**4
n_ref = 9

dim, L2err, hmin, time = uniform_refinement(N, n_ref)

folder_name = '/uniform'
path_folder = path + folder_name

if not(os.path.exists(path_folder)):
    os.makedirs(path_folder)
np.savez(path_folder + f"{folder_name}.npz",
         dim=dim, L2err=L2err, hmin=hmin, time=time)
