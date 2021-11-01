#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:00:09 2020


Solve Equidistribution mesh in 1D for H1 function 

# Use MMPDE5 until equidistribution
# Find optimal mesh density function 
# Find best parameter alpha

# Piecewise linear elements evaluated in L2 norm

@author: simone
"""

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit

Tol = 1E-14
parameters['allow_extrapolation'] = True


def boundary_left(x, on_boundary):
    return on_boundary and near(x[0], 0, Tol)


def boundary_right(x, on_boundary):
    return on_boundary and near(x[0], 1.0, Tol)


# Expression for exact solution
class Expression_uexact(UserExpression):

    def __init__(self, exponent, rho, **kwargs):
        super().__init__(**kwargs)  # This part is new!
        self.exp = exponent
        self.rho = rho

    def eval(self, value, x):

        if x[0] > 0 and x[0] < 0.5*self.rho:

            value[0] = pow(x[0], self.exp)

        elif x[0] >= 0.5*self.rho and x[0] <= self.rho:

            value[0] = (15.0/16.0)*(8.0/15.0 - (4*x[0]/self.rho - 3) + 2.0/3.0*pow(4*x[0] /
                                                                                   self.rho - 3, 3) - 1.0/5.0*pow(4.0*x[0]/self.rho - 3, 5))*pow(x[0], self.exp)

        else:

            value[0] = 0.0

    def value_shape(self):
        return ()


def solve_poisson(V, mesh):

    v = TestFunction(V)
    u = TrialFunction(V)

    bcs = [DirichletBC(V, 0.0, boundary_left),
           DirichletBC(V, u_expr, boundary_right)]

    a = inner(grad(v), grad(u))*dx(domain=mesh)
    L = f*v*dx(domain=mesh)

    u = Function(V)
    solve(a == L, u, bcs)

    return u


def a_priori_bound(mesh):

    h = CellDiameter(mesh)
    w = TestFunction(DG0)
    x = SpatialCoordinate(mesh)
    a_priori_bound = Function(DG0)

    bound = w*pow(h, 7.0/3.0)*(1.0/x[0]) * \
        math.gamma(5.0/3.0)/math.gamma(0.5)*dx(mesh)
    assemble(bound, tensor=a_priori_bound.vector())

    bound = np.sum(a_priori_bound.vector()[:])

    return bound


def monitor_posteriori(mesh):

    diam = CellDiameter(mesh)
    x = mesh.coordinates()[:, 0]
    w = TestFunction(DG0)
    avg_residual_squared = Function(DG0)

    # assume avg diameter close to edge length
    residual = w*(1/diam)*(u_post.dx(0).dx(0)+f) * \
        (u_post.dx(0).dx(0)+f)*dx(mesh)
    assemble(residual, tensor=avg_residual_squared.vector())

    #alpha = 1
    alpha_h = np.sum(
        np.diff(x)*np.power(avg_residual_squared.vector()[:], 1.0/3.0))**3
    avg_residual_squared = interpolate(avg_residual_squared, V1)

    # w must be interpolated in the computational mesh for integration
    w_post = Function(V1)
    w_post.vector()[:] = np.power(
        (1 + (1/alpha_h)*avg_residual_squared.vector()[:]), 1.0/3.0)

    return w_post


def monitor_interpolant(mesh, u_ex):

    diam = CellDiameter(mesh)
    x = mesh.coordinates()[:, 0]
    w = TestFunction(DG0_2)
    avg_u_squared = Function(DG0_2)

    ux = project(u_ex.dx(0), V2)
    uxx = project(ux.dx(0), V2)
    # assume avg diameter close to edge length
    u_form = w*(1/diam)*(uxx)*(uxx)*dx(mesh)
    assemble(u_form, tensor=avg_u_squared.vector())

    #alpha = 1.0
    alpha_h = np.sum(
        np.diff(x)*np.power(avg_u_squared.vector()[:], 1.0/5.0))**5
    avg_u_squared = interpolate(avg_u_squared, V2)

    # w must be interpolated in the computational mesh for integration
    w_interp = Function(V2)
    w_interp.vector()[:] = np.power(
        (1 + (1/alpha_h)*avg_u_squared.vector()[:]), 1.0/5.0)

    return w_interp


# smoothing gives smoother increment of the local skewness
def smoothing(w, V):

    w_test = TestFunction(V)
    w_trial = TrialFunction(V)

    a = w_test*w_trial*dx + 1e-8*inner(grad(w_test), grad(w_trial))*dx
    L = w_test*w*dx

    w = Function(V)
    solve(a == L, w)

    # scale
    w.vector()[:] = w.vector()[:]/np.max(w.vector()[:])

    return w


def equidistribute(x, w, dof2vertex_map):

    rho = w.vector()[dof2vertex_map]
    # Make a copy of vector x to avoid overwriting
    y = x.copy()

    # number of mesh points counted from 0 ton nx-1
    nx = x.shape[0]

    II = nx - 1
    JJ = nx - 1

    # Create vector of integrals with nx entries
    intMi = np.zeros(nx)

    # compute each integral using trapezoidal rule
    intMi[1:] = 0.5*(rho[1:] + rho[:-1])*np.diff(x)

    # take cumulative sum of integrals
    intM = np.cumsum(intMi)

    # take total integral theta
    theta = intM[-1]

    jj = 0

    # Assign new nodes from  y_1 to y_(nx - 2)
    for ii in range(1, II):

        # Target =  y_1 = 1/(nx-1)*theta ... y_nx-2 = (nx-2)/(nx-1)*theta

        Target = ii/II*theta

        while jj < JJ and intM[jj] < Target:

            jj = jj+1

        jj = jj - 1

        Xl = x[jj]
        Xr = x[jj+1]
        Ml = rho[jj]
        Mr = rho[jj+1]

        Target_loc = Target - intM[jj]

        mx = (Mr - Ml)/(Xr - Xl)

        y[ii] = Xl + 2*Target_loc/(Ml + np.sqrt(Ml**2 + 2*mx*Target_loc))

        y[0] = 0.0
        y[-1] = 1.0

    return y


def conv_rate(dof, err):
    'Compute convergence rate '

    l = dof.shape[0]
    rate = np.zeros(l-1)

    for i in range(l-1):
        rate[i] = ln(np.sqrt(err[i]/err[i+1]))/ln(dof[i+1]/dof[i])

    return rate


Ns = 2**(np.arange(4, 13))


post_hmin = np.zeros(len(Ns))
interp_hmin = np.zeros(len(Ns))

L2err_interp = np.zeros(len(Ns))
L2err_post = np.zeros(len(Ns))

dim_interp = np.zeros(len(Ns))
dim_post = np.zeros(len(Ns))

time_post = np.zeros(len(Ns))
time_interp = np.zeros(len(Ns))

ratio_interp = np.zeros(Ns.shape[0])
ratio_post = np.zeros(Ns.shape[0])

abstol = 1e-16
max_iter = 20

for it, N in enumerate(Ns):

    mesh1 = UnitIntervalMesh(N)  # mesh a-posteriori error
    mesh2 = UnitIntervalMesh(N)  # mesh interpolation error estimate

    DG0 = FunctionSpace(mesh1, 'DG', 0)
    DG0_2 = FunctionSpace(mesh2, 'DG', 0)

    V1 = FunctionSpace(mesh1, 'CG', 1)
    V2 = FunctionSpace(mesh2, 'CG', 1)

    E1 = FunctionSpace(mesh1, 'CG', 5)
    E2 = FunctionSpace(mesh2, 'CG', 5)

    #u_expr = Expression_uexact(exponent= 2.0/3.0 ,rho= 1.0,degree=5)
    u_expr = Expression('pow(x[0],2.0/3.0)', degree=5)
    u_ex1 = Function(E1)
    u_ex2 = Function(E2)

    u_ex1.interpolate(u_expr)
    u_ex2.interpolate(u_expr)

    dof2vertex_map1 = dof_to_vertex_map(V1)
    dof2vertex_map2 = dof_to_vertex_map(V2)

    # Solve 1D Poisson equation
    f = -div(grad(u_ex1))
    u_post = solve_poisson(V1, mesh1)

    tol = 1.0
    iteration = 0

    # equidistribute mesh with optimal functions given by a-posteriori error until specific tolerance is reached
    print('Equidistribution by a-posteriori error upper bound')

    t0 = timeit.default_timer()
    while (tol > abstol) and (iteration < max_iter):

        print('rel_tol is: ', tol)
        # update monitor function

        x1 = mesh1.coordinates()[:, 0]
        w_post = monitor_posteriori(mesh1)
        #w_post = smoothing(w_post,V1,mesh1)

        x_1 = equidistribute(x1, w_post, dof2vertex_map1)
        tol = max(abs(x1 - x_1))

        # solve Poisson eq in new mesh to define w_post
        mesh1.coordinates()[:, 0] = x_1

        u_ex1.interpolate(u_expr)
        f = -div(grad(u_ex1))
        u_post = solve_poisson(V1, mesh1)

        iteration += 1

    time_post[it] = timeit.default_timer() - t0

    print('Equidistribution by interpolation error upper bound')
    tol = 1.0
    iteration = 0

    u_ex2.interpolate(u_expr)
    f = -div(grad(u_ex2))
    u_interp = solve_poisson(V2, mesh2)

    t0 = timeit.default_timer()

    while (tol > abstol) and (iteration < max_iter):

        w_interp = monitor_interpolant(mesh2, u_interp)

        x2 = mesh2.coordinates()[:, 0]
        x_2 = equidistribute(x2, w_interp, dof2vertex_map2)

        tol = max(abs(x2 - x_2))
        mesh2.coordinates()[:, 0] = x_2

        u_ex2 = interpolate(u_expr, E2)
        f = -div(grad(u_ex2))
        u_interp = solve_poisson(V2, mesh2)

        iteration += 1

    time_interp[it] = timeit.default_timer() - t0

    # compute errors in L2 norm
    dx = Measure('dx', domain=mesh1)
    L2err_post[it] = assemble((u_expr*u_expr + u_post*u_post - 2*u_expr*u_post)
                              * dx(mesh1), form_compiler_parameters={"quadrature_degree": 5})
    post_hmin[it] = min(abs(np.diff(np.sort(x_1))))
    dim_post[it] = V2.dim()

    dx = Measure('dx', domain=mesh2)
    L2err_interp[it] = assemble((u_expr*u_expr + u_interp*u_interp - 2*u_expr*u_interp)
                                * dx(mesh2), form_compiler_parameters={"quadrature_degree": 5})
    dim_interp[it] = V2.dim()
    interp_hmin[it] = min(abs(np.diff(np.sort(x_2))))


def refinement(mesh, u, ref_ratio=False, tol=1.0):

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


N = 2**4
n_ref = 25

mesh = UnitIntervalMesh(N)  # mesh a-posteriori error

href_hmin = np.zeros(n_ref)
L2err_href = np.zeros(n_ref)
dim_href = np.zeros(n_ref)

time_href = np.zeros(n_ref)
t0 = timeit.default_timer()

for it in range(n_ref):

    DG0 = FunctionSpace(mesh, 'DG', 0)
    V = FunctionSpace(mesh, 'CG', 1)
    E = FunctionSpace(mesh, 'CG', 5)
    u_ex = Function(E)

    # Solve 1D Poisson equation
    u_ex.interpolate(u_expr)
    f = -div(grad(u_ex))
    u_ref = solve_poisson(V, mesh)

    dx = Measure('dx', domain=mesh)
    L2err_href[it] = assemble((u_expr*u_expr + u_ref*u_ref - 2*u_expr*u_ref)
                              * dx(mesh1), form_compiler_parameters={"quadrature_degree": 5})
    dim_href[it] = V.dim()

    mesh = refinement(mesh, u_ref, ref_ratio=0.25)
    time_href[it] = timeit.default_timer() - t0
    x_3 = mesh.coordinates()[:, 0]
    dof2vertex_map3 = dof_to_vertex_map(V)
    href_hmin[it] = min(np.diff(np.sort(x_3)))


N = 2**4
n_ref = 9

mesh = UnitIntervalMesh(N)

L2err_unif = np.zeros(n_ref)
dim_unif = np.zeros(n_ref)
unif_hmin = np.zeros(n_ref)


time_unif = np.zeros(n_ref)
t0 = timeit.default_timer()

for it in range(n_ref):

    DG0 = FunctionSpace(mesh, 'DG', 0)
    V = FunctionSpace(mesh, 'CG', 1)
    E = FunctionSpace(mesh, 'CG', 5)
    u_ex = Function(E)

    # Solve 1D Poisson equation
    u_ex.interpolate(u_expr)
    f = -div(grad(u_ex))
    u_ref = solve_poisson(V, mesh)

    dx = Measure('dx', domain=mesh)
    L2err_unif[it] = assemble((u_expr*u_expr + u_ref*u_ref - 2*u_expr*u_ref)
                              * dx(mesh), form_compiler_parameters={"quadrature_degree": 5})
    dim_unif[it] = V.dim()

    mesh = refine(mesh)
    time_unif[it] = timeit.default_timer() - t0

    x_3 = mesh.coordinates()[:, 0]
    dof2vertex_map3 = dof_to_vertex_map(V)
    unif_hmin[it] = min(np.diff(np.sort(x_3)))


# Plot CPU execution time for different mesh adaptation algorithms

plt.figure()
plt.plot(dim_unif, time_unif, linestyle='-.', marker='o', label='uniform')
plt.plot(dim_interp, time_interp, linestyle='-.',
         marker='x', label='interpolation')
plt.plot(dim_post, time_post, linestyle='-.', marker='v', label='a-posteriori')
plt.plot(dim_href, time_href, linestyle='-.', marker='^', label='h-ref')
plt.xlabel('N')
plt.ylabel('CPU execution time')
plt.yscale('log')
plt.xscale('log')
plt.legend()


rate_unif = conv_rate(dim_unif, L2err_unif)
rate_interp = conv_rate(dim_interp, L2err_interp)
rate_post = conv_rate(dim_post, L2err_post)
rate_href = conv_rate(dim_href, L2err_href)

plt.figure()
plt.plot(dim_unif, L2err_unif, linestyle='-.', marker='o',
         label='uniform | rate %4g' % rate_unif[-1])
plt.plot(dim_interp, L2err_interp, linestyle='-.', marker='x',
         label='interpolation | rate %4g' % rate_interp[-1])
plt.plot(dim_post, L2err_post, linestyle='-.', marker='v',
         label='a-posteriori | rate %4g' % np.mean(rate_post))
plt.plot(dim_href, L2err_href, linestyle='-.', marker='^',
         label='h-ref | rate %4g' % np.mean(rate_href[-10:]))
plt.xlabel('N')
plt.ylabel('L2-error')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()

plt.figure()
plt.plot(unif_hmin, linestyle='-.', marker='o', label='uniform')
plt.plot(interp_hmin, linestyle='-.', marker='x', label='interpolation')
plt.plot(post_hmin, linestyle='-.', marker='v', label='a-posteriori')
plt.plot(href_hmin, linestyle='-.', marker='^', label='h-ref')
plt.ylabel('min($\Delta x$)')
plt.xlabel('iteration')
plt.yscale('log')
plt.xscale('log')
plt.legend()


# np.save('L2_time_unif'+ '1D.npy',time_unif)
# np.save('L2_time_interp'+ '1D.npy',time_interp)
# np.save('L2_time_post'+ '1D.npy',time_post)
# np.save('L2_time_href'+ '1D.npy',time_href)


# np.save('L2_unif'+ '1D.npy',L2err_unif)
# np.save('L2_interp'+ '1D.npy',L2err_interp)
# np.save('L2_post'+  '1D.npy',L2err_post)
# np.save('L2_href'+ '1D.npy',L2err_href)


# np.save('dof_L2unif'+  '1D.npy',dim_unif)
# np.save('dof_L2interp'+  '1D.npy',dim_interp)
# np.save('dof_L2post'+ '1D.npy',dim_post)
# np.save('dof_L2href'+ '1D.npy',dim_href)


# plot solution with mesh interpolant and a-posteriori
# plt.figure()
# plt.scatter(x_1, u_post.vector()[dof2vertex_map1],
#             marker='v', label='a-posteriori')
# plt.scatter(x_2, u_interp.vector()[
#             dof2vertex_map2], marker='x', label='interpolation')
# plt.plot(np.sort(x_3)[::-1], u_ex.vector()[:], label='exact')
# plt.xlabel('x')
# plt.ylabel('u')
# plt.legend()
