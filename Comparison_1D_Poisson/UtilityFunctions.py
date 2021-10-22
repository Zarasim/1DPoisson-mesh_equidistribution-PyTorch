
import math
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
parameters['allow_extrapolation'] = True


tol = 1E-14


def boundary_left(x, on_boundary):
    return on_boundary and near(x[0], 0, tol)


def boundary_right(x, on_boundary):
    return on_boundary and near(x[0], 1.0, tol)


# Expression for exact solution
class ExpressionPoisson(UserExpression):

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


def solve_poisson(V, mesh, u_expr, f):

    v = TestFunction(V)
    u = TrialFunction(V)

    bcs = [DirichletBC(V, 0.0, boundary_left),
           DirichletBC(V, u_expr, boundary_right)]

    a = inner(grad(v), grad(u))*dx(domain=mesh)
    L = f*v*dx(domain=mesh)

    u = Function(V)
    solve(a == L, u, bcs)

    return u


def a_priori_bound(mesh, DG0):

    h = CellDiameter(mesh)
    w = TestFunction(DG0)
    x = SpatialCoordinate(mesh)
    a_priori_bound = Function(DG0)

    bound = w*pow(h, 7.0/3.0)*(1.0/x[0]) * \
        math.gamma(5.0/3.0)/math.gamma(0.5)*dx(mesh)
    assemble(bound, tensor=a_priori_bound.vector())

    bound = np.sum(a_priori_bound.vector()[:])

    return bound


def monitor_posteriori(mesh, DG0, V1, u_post, f):

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


def monitor_interpolant(mesh, DG0, V, u_ex):

    diam = CellDiameter(mesh)
    x = mesh.coordinates()[:, 0]
    w = TestFunction(DG0)
    avg_u_squared = Function(DG0)

    ux = project(u_ex.dx(0), V)
    uxx = project(ux.dx(0), V)
    # assume avg diameter close to edge length
    u_form = w*(1/diam)*(uxx)*(uxx)*dx(mesh)
    assemble(u_form, tensor=avg_u_squared.vector())

    #alpha = 1.0
    alpha_h = np.sum(
        np.diff(x)*np.power(avg_u_squared.vector()[:], 1.0/5.0))**5
    avg_u_squared = interpolate(avg_u_squared, V)

    # w must be interpolated in the computational mesh for integration
    w_interp = Function(V)
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
    """Compute convergence rate """

    l = dof.shape[0]
    rate = np.zeros(l-1)

    for i in range(l-1):
        rate[i] = ln(np.sqrt(err[i]/err[i+1]))/ln(dof[i+1]/dof[i])

    return rate
