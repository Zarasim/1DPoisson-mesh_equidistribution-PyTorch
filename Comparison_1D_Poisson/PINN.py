import torch
from torch import nn
import numpy as np
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev, splder

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
    #eps = 1e-9*torch.randn_like(x)
    #x += eps
    if (len(x) != len(set(x))):
        print('collision in DNN rescale')
    x = torch.sort(x)[0]
    # Avoid to have to equal coordinates, add a random eps to each variable
    y = x - torch.min(x)
    z = y.squeeze()/(torch.max(y).squeeze())
    z = lb + (ub - lb)*z

    return z

# the deep neural network


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
        out = rescale(out, self.lb, self.ub)
        return out

    def save(self, path_model, iter):
        # Python dictionary object that maps each layer to its parameter tensor.
        torch.save(self.state_dict(), path_model + f'/DNN_{iter}.pth')

    def load(self, path):
        # load the parameters from pth file
        self.load_state_dict(torch.load(path))


def init_weights(m):
    with torch.no_grad():
        # 0.01 for this time
        if type(m) == torch.nn.Linear:
            m.weight.normal_(1, 0.1)
            m.bias.normal_(0, 0.1)


class PhysicsInformedNN():

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
        self.X_prev = np.linspace(0, 1, self.N)
        self.u = u
        self.k = k
        self.m = m
        self.relErr = relErr
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

    def avg_derivative(self, h, x, u, X_prev):
        """ Compute first/second order derivative of u"""

        ## compute analitycally the first derivative  ###
        # for i,v in enumerate(x):
        #   u[i] = 10*mp.sech(10*v)**2

        ## compute analitycally the second derivative ###
        # for i,v in enumerate(x):
        #    u[i] = -200*(mp.sech(10*v + 1e-6)**2)*np.tanh(10*v)
        # print(x)
        X = x.detach().cpu().numpy()
        # print(X)
        #X = rescale_(X, self.lb, self.ub, self.iter)
        X = np.array(sorted(X))

        H = h.detach().cpu().numpy()
        H = sorted(H)

        Y = u.detach().cpu().numpy()
        iterc = 0
        if not np.all(np.diff(X) >= 0.0):
            # find indexes of colliding points
            #idxs = np.where(np.diff(X) == 0.0)
            # print(np.diff(X))
            # print(X)
            print('colliding points')
            X = X_prev
            #rescale_(X, self.lb, self.ub, iterc)

        u_interp = IUS(X, Y)
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

        return res, X

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
        u = self.u(x)
        u_xx, self.X_prev = self.avg_derivative(h, x, u, self.X_prev)
        #alpha = self.optimal_alpha(h,u_xx)
        alpha = 1
        f = h*torch.pow(1 + 1/alpha*u_xx, 1/(1+2*(self.k-self.m+1)))
        f = torch.pow(f, 1+2*(self.k-self.m+1))

        return f, alpha

    def loss_func(self):

        self.optimizer.zero_grad()
        f, alpha = self.net_equid(self.N, self.xi)
        Loss = ((self.N-1)**(2*(self.k-self.m+1)))*alpha*torch.sum(f)

        if Loss > 0:
            self.loss.append(Loss)
            Loss.backward()
        print('Iter %d, Loss: %.5e' % (self.iter, Loss.item()))
        return Loss

    def get_loss(self):
        return self.loss

    def train(self):

        while self.iter < self.max_it:

            self.dnn.train()
            self.iter += 1
            # Backward and optimize
            self.optimizer.step(self.loss_func)

            relErr = 100
            if self.iter > 2:
                relErr = np.abs(self.loss[-1].item() -
                                self.loss[-2].item())/(self.loss[-2].item())

            if self.iter % 500 == 0:
                self.dnn.save(self.path_model, self.iter)

            if relErr < self.relErr:
                self.dnn.save(self.path_model, self.iter)
                return

        self.dnn.save(self.path_model, self.iter)
        return

    def predict(self, Xi):

        Xi = torch.tensor(Xi, requires_grad=True).float(
        ).unsqueeze(-1).to(self.device)
        self.dnn.eval()
        p = self.dnn(Xi)
        p = rescale(p, self.lb, self.ub)
        return p


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
