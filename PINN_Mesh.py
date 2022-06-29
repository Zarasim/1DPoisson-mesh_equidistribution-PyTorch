import torch
from torch import nn
import numpy as np
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d
import os


from dolfin import *


class softplus_power(torch.nn.Module):
    def __init(self):
        super.__init__()

    def forward(self, x):
        m = nn.Softplus()
        x = m(x)
        return x**1.1


def rescale_(x, lb, ub):
    """ rescaling function to impose boundary conditions """

    lb = lb.detach().cpu().numpy()
    ub = ub.detach().cpu().numpy()
    if (len(x) != len(set(x))):
        eps = 10**(-8)*np.random.randn(len(x))
        print('collision in rescale')
        x += 10**5*eps
    x = sorted(x)
    # Avoid to have to equal coordinates, add a random eps to each variable
    y = x - np.min(x)
    z = y/np.max(y)
    z = lb + (ub - lb)*z

    return z


def rescale(x, lb, ub):
    """ rescaling function to impose boundary conditions """
    if (len(x) != len(set(x))):
        print('collision in DNN rescale')
    x = torch.sort(x)[0]
    # Avoid to have to equal coordinates, add a random eps to each variable
    y = x - torch.min(x)
    z = y.squeeze()/(torch.max(y).squeeze())
    z = lb + (ub - lb)*z

    return z


# The deep neural network
class DNN_Mesh(torch.nn.Module):

    # layers count the hiddens + the output
    def __init__(self, layers, lb, ub):

        super().__init__()

        # parameters
        self.depth = len(layers) - 1
        self.lb = lb
        self.ub = ub
        self.activation = nn.LeakyReLU()
        #self.activation = nn.Softplus()
        self.time = 0
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
        out = rescale(out, self.lb, self.ub)
        return out

    def save(self, path_model, iter):
        # Python dictionary object that maps each layer to its parameter tensor.
        torch.save(self.state_dict(), path_model + f'/DNN/DNN_{iter}.pth')

    def load(self, path):
        # load the parameters from pth file
        self.load_state_dict(torch.load(path))


def init_weights(m):
    with torch.no_grad():
        if type(m) == torch.nn.Linear:
            m.weight.normal_(1, 0.1)
            m.bias.normal_(0, 0.1)


class PINN_Mesh():

    def __init__(self, N, xi, u, layers, lb, ub, device, path_model, max_it=1000, k=1, m=0, lr=0.1, relErr=1e-8, net=[]):

        # boundary conditions
        self.device = device
        self.lb = torch.tensor(lb).float().to(self.device)
        self.ub = torch.tensor(ub).float().to(self.device)

        # data
        self.N = N
        self.xi = torch.tensor(xi, requires_grad=True).float().to(
            self.device).unsqueeze(-1)
        self.layers = layers
        self.u = u
        self.k = k
        self.m = m
        self.relErr = relErr
        self.max_it = max_it
        self.path_model = path_model
        # neural net
        self.loss = []
        self.lr = lr

        if net == []:
            self.dnn = DNN_Mesh(self.layers, self.lb, self.ub).to(self.device)
            # manual_seed works only with CUDA
            torch.manual_seed(1234)
            self.dnn.apply(init_weights)
        else:
            print('net loaded')
            self.dnn = net

        # works well when the first/second order derivative is approximated
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=self.lr)
        self.iter = 0

    def avg_derivative(self, h, x, X_u, u):
        """ Compute first/second order derivative of u"""

        X = x.detach().cpu().numpy()
        X = np.array(sorted(X))

        H = h.detach().cpu().numpy()
        H = sorted(H)

        #Y = u.detach().cpu().numpy()
        u_interp = IUS(X_u, u)
        u_x = u_interp.derivative()
        f = u_x.derivative()

        # integrate f numerically for each subinterval
        Integral = np.zeros_like(H)
        for i in range(len(H)):
            Integral[i] = integrate.quad(lambda x: f(x)**2, X[i], X[i+1])[0]

        res = np.zeros_like(H)
        for k, I in enumerate(H):
            res[k] = (1/(I+1e-6))*Integral[k]

        res = torch.tensor(res).float().to(self.device)

        return res

    # Not needed in practice
    def optimal_alpha(self, h, u_xx):

        Sum = torch.sum(h*u_xx**(1/(1+2*(self.k-self.m+1))))
        print('SUM is: ', Sum)
        alpha = torch.pow(1/(self.ub - self.lb)*Sum, 1+2*(self.k-self.m+1))

        return alpha

    def net_equid(self, N, xi):

        x = self.dnn(xi)
        h = x.squeeze()
        h = torch.sort(h)[0]
        h = h[1:] - h[:-1]
        u_xx = self.avg_derivative(h, x, self.X_u, self.u)
        alpha = 1
        f = h*torch.pow(1 + 1/alpha*u_xx, 1/(1+2*(self.k-self.m+1)))
        f = torch.pow(f, 1+2*(self.k-self.m+1))

        return f, alpha

    def loss_func(self):

        self.optimizer.zero_grad()
        f, alpha = self.net_equid(self.N, self.xi)
        loss = ((self.N-1)**(2*(self.k-self.m+1)))*alpha*torch.sum(f)

        self.loss.append(loss.item())
        loss.backward()
        print('Iter %d, Loss: %.5e' % (self.iter, loss.item()))
        return loss

    def get_loss(self):
        return self.loss

    def train(self, ep, X, u):

        self.X_u = X
        self.u = u
        os.makedirs(self.path_model + '/DNN', exist_ok=True)
        os.makedirs(self.path_model + '/Loss', exist_ok=True)
        while self.iter < self.max_it*ep:

            self.dnn.train()
            self.iter += 1
            # Backward and optimize
            self.optimizer.param_groups[0]['lr'] = self.lr * \
                10**(-np.log10(1+self.iter/100))

            self.optimizer.step(self.loss_func)

            # relErr = 100
            # if self.iter > 2:
            #     relErr = np.abs(self.loss[-1]-self.loss[-2])/(self.loss[-2])

            if self.iter % (self.max_it//2) == 0:
                self.dnn.save(self.path_model, self.iter)
                np.save(self.path_model +
                        f'/Loss/loss_{self.iter}.npy', self.loss)

            # if relErr < self.relErr:
            #     self.dnn.save(self.path_model, self.iter)
            #     np.save(self.path_model +
            #             f'/Loss/loss_{self.iter}.npy', self.loss)
            #     return

        self.dnn.save(self.path_model, self.iter)
        np.save(self.path_model + f'/Loss/loss_{self.iter}.npy', self.loss)
        return

    def predict(self, Xi):

        Xi = torch.tensor(Xi, requires_grad=True).float(
        ).unsqueeze(-1).to(self.device)
        self.dnn.eval()
        p = self.dnn(Xi)
        p = rescale(p, self.lb, self.ub)
        p = p.detach().cpu().squeeze().numpy()
        return p


def convRate(model, a, b, Nvec, U, equid=True):
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


# def convRate(model, a, b, Nvec, u, equid=True):
#     Errvec = np.zeros_like(Nvec)
#     rate = np.zeros_like(Nvec[:-1])
#     for i, N in enumerate(Nvec):
#         x = np.linspace(a, b, N)
#         if not equid:
#             x = model.predict(x).detach().cpu().numpy()
#         y = u(x)
#         f_interp = interp1d(x, y)
#         err = integrate.quad(lambda x: (u(x) - f_interp(x))
#                              ** 2, a, b, epsabs=0)[0]
#         Errvec[i] = np.sqrt(err)
#     rate = np.log(Errvec[:-1]/Errvec[1:])/np.log(Nvec[1:]/Nvec[:-1])
#     return rate, Errvec
