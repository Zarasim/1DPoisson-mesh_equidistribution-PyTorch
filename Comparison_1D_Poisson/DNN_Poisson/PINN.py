import torch
from torch import nn
import numpy as np
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev, splder
from dolfin import *
from Comparison_1D_Poisson.DNN_Mesh_Poisson.trainDNN import X
from Comparison_1D_Poisson.equidMesh import L2err


# The deep neural network
class DNN(torch.nn.Module):

    # layers count the hiddens + the output
    def __init__(self, layers, lb, ub):

        super().__init__()

        # parameters
        self.depth = len(layers) - 1
        self.lb = lb
        self.ub = ub
        self.activation = nn.LeakyReLU()
        #self.activation = nn.Softplus()

        # set up layer order dict
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        for i in range(self.depth - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
            self.activations.append(self.activation)

        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):

        out = x
        for i in range(len(self.layers)-1):
            out = self.layers[i](out)
            out = self.activations[i](out)

        # out is of shape (ndata,1)
        out = self.layers[-1](out)
        out = fixboundary(out, self.lb, self.ub)
        return out

    def save(self, path_model, iter):
        # Python dictionary object that maps each layer to its parameter tensor.
        torch.save(self.state_dict(), path_model + f'/DNN_{iter}.pth')

    def load(self, path):
        # load the parameters from pth file
        self.load_state_dict(torch.load(path))


class PINN():

    # Solve 1D Poisson equation using Fenics with DB conditions

    def __init__(self, mesh, f, layers, boundary, device, path_model, max_it=1000, lr=0.1, relErr=1e-8, net=[]):

        # boundary conditions
        self.device = device
        self.bl = boundary[0]
        self.br = boundary[1]
        self.layers = layers
        self.f = f
        self.relErr = relErr
        self.mesh = mesh
        self.DG0 = FunctionSpace(self.mesh, 'DG', 0)
        self.CG1 = FunctionSpace(self.mesh, 'CG', 1)
        self.CG5 = FunctionSpace(self.mesh, 'CG', 5)
        self.u_ex = Function(self.CG5)
        self.u_expr = Expression('pow(x[0],2.0/3.0)', degree=5)

        # neural net
        self.loss = []
        self.lr = lr

        if net == []:
            self.dnn = DNN(self.layers, self.lb, self.ub).to(self.device)
            # manual_seed works only with CUDA
            torch.manual_seed(1234)
            self.dnn.apply(init_weights)
        else:
            print('net loaded')
            self.dnn = net

        self.max_it = max_it
        self.path_model = path_model

        # optimizers: using the same settings
        #self.optimizer = torch.optim.RMSprop(self.dnn.parameters(), lr=1e-5)
        # works well when the first/second order derivative is computed analitycally
        #self.optimizer = torch.optim.Adadelta(self.dnn.parameters(), lr=0.1)
        # works well when the first/second order derivative is approximated
        # 0.1 default
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=self.lr)
        #self.optimizer = torch.optim.LBFGS(self.dnn.parameters(), lr=0.1)
        self.iter = 0

    def solve_poisson(self, V, mesh):

        v = TestFunction(V)
        u = TrialFunction(V)

        bcs = [DirichletBC(V, 0.0, self.bl),
               DirichletBC(V, 1.0, self.br)]

        a = inner(grad(v), grad(u))*dx(domain=mesh)
        L = self.f*v*dx(domain=mesh)

        u = Function(V)
        solve(a == L, u, bcs)

        return u

    def loss_func(self):

        self.optimizer.zero_grad()
        u = self.dnn(self.x)

        X = self.x.detach().cpu().numpy()
        U = self.u.detach().cpu().numpy()
        u_interp = IUS(X, U)
        u_x = u_interp.derivative()
        f_pred = u_x.derivative()
        Loss = (self.f - f_pred)**2

        self.loss.append(Loss)
        Loss.backward()
        print('Iter %d, Loss: %.5e' % (self.iter, Loss.item()))
        return Loss

    def get_loss(self):
        return self.loss

    def train(self):
        # Construct update mesh with new coordinates
        while self.iter < self.max_it:

            self.dnn.train()
            self.iter += 1
            # Backward and optimize
            self.optimizer.step(self.loss_func)

            if self.iter > 2:
                relErr = np.abs(self.loss[-1].item() -
                                self.loss[-2].item())/(self.loss[-2].item())
            else:
                relErr = 1

            if self.iter % 500 == 0:
                self.dnn.save(self.path_model, self.iter)

            if relErr < self.relErr:
                self.dnn.save(self.path_model, self.iter)
                return

        self.dnn.save(self.path_model, self.iter)
        return

    def predict(self, X):
        X = torch.tensor(X, requires_grad=True).float(
        ).unsqueeze(-1).to(self.device)
        self.dnn.eval()
        u = self.dnn(X)
        u.detach().cpu().numpy()
        return u


def fixboundary(u, bl, br):
    u[0] = bl
    u[-1] = br


def convRate(model, a, b, Nvec, u, equid=True):
    Errvec = np.zeros_like(Nvec)
    rate = np.zeros_like(Nvec[:-1])
    for i, N in enumerate(Nvec):
        x = np.linspace(a, b, N)
        if not equid:
            x = model.predict(x).detach().cpu().numpy()
        y = u(x)
        f_interp = interp1d(x, y)
        err = integrate.quad(lambda x: (u(x) - f_interp(x))
                             ** 2, a, b, epsabs=0)[0]
        Errvec[i] = np.sqrt(err)

    rate = np.log(Errvec[:-1]/Errvec[1:])/np.log(Nvec[1:]/Nvec[:-1])
    return rate, Errvec


def convRate_fenics(model, a, b, Nvec, U, equid=True):
    Errvec = np.zeros_like(Nvec)
    rate = np.zeros_like(Nvec[:-1])

    for i, N in enumerate(Nvec):
        x = np.linspace(a, b, int(N))
        mesh = UnitIntervalMesh(int(N)-1)
        if not equid:
            x = model.predict(x).detach().cpu().numpy()
        mesh.coordinates()[:, 0] = x
        CG1 = FunctionSpace(mesh, 'CG', 1)
        dof2vertex_map = dof_to_vertex_map(CG1)
        u_expr = Expression('pow(x[0],2.0/3.0)', degree=5)
        u = Function(CG1)
        Uvec = np.array([U(x[i]) for i in range(len(x))])
        u.vector()[dof2vertex_map] = Uvec

        err = assemble((u_expr*u_expr + u*u - 2*u_expr*u)
                       * dx(mesh), form_compiler_parameters={"quadrature_degree": 5})
        Errvec[i] = np.sqrt(err)

    rate = np.log(Errvec[:-1]/Errvec[1:])/np.log(Nvec[1:]/Nvec[:-1])
    return rate, Errvec


def init_weights(m):
    with torch.no_grad():
        if type(m) == torch.nn.Linear:
            m.weight.normal_(1, 0.1)
            m.bias.normal_(0, 0.1)
