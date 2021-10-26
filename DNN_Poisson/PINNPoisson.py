from genericpath import exists
import torch
from torch import nn
import numpy as np
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d
#from scipy.interpolate import splrep, splev, splder
from dolfin import *
from collections import OrderedDict
import time
import os


class softplus_power(nn.Module):
    def __init(self):
        super.__init__()

    def forward(self, x):
        m = nn.Softplus()
        x = m(x)
        return x**1.2


# The deep neural network
class DNN(torch.nn.Module):

    # layers count the hiddens + the output
    def __init__(self, layers):

        super().__init__()

       # parameters
        self.depth = len(layers) - 1
        self.dropout = torch.nn.Dropout(0.1)
        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1),
             torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

    def save(self, path_model, iter):
        # Python dictionary object that maps each layer to its parameter tensor.
        torch.save(self.state_dict(), path_model + f'/DNN_{iter}.pth')

    def load(self, path):
        # load the parameters from pth file
        self.load_state_dict(torch.load(path))


class PINN():

    # Solve 1D Poisson equation using Fenics with DB conditions

    def __init__(self, X, f, layers, ul, ur, device, path_model, batch_size=100, max_it=1000, lr=0.1, relErr=1e-8, net=[]):

        self.device = device
        self.ul = torch.tensor(ul).float().to(self.device)
        self.ur = torch.tensor(ur).float().to(self.device)
        self.x = torch.tensor(X, requires_grad=True).float().to(
            self.device).unsqueeze(-1)

        self.N = len(X)
        self.layers = layers
        self.f = torch.tensor(f(X)).to(self.device)
        self.batch_size = batch_size

        self.relErr = relErr

        # neural net
        self.loss = []
        self.lr = lr

        if net == []:
            print('net created')
            self.dnn = DNN(self.layers).to(self.device)
            # manual_seed works only with CUDA
            torch.manual_seed(1234)
            # self.dnn.apply(init_weights)
        else:
            print('net loaded')
            self.dnn = net

        self.max_it = max_it
        self.path_model = path_model
        self.iter = 0
        # optimizers: using the same settings
        # self.optimizer = torch.optim.RMSprop(self.dnn.parameters(), lr=1e-5)
        # works well when the first/second order derivative is computed analitycally
        #self.optimizer = torch.optim.Adadelta(self.dnn.parameters(), lr=0.1)
        # works well when the first/second order derivative is approximated
        # 0.1 default
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=self.lr)
        #self.optimizer = torch.optim.SGD(self.dnn.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.LBFGS(
        #     self.dnn.parameters(),
        #     lr=self.lr,
        #     max_iter=50000,
        #     max_eval=50000,
        #     history_size=50,
        #     tolerance_grad=1e-5,
        #     tolerance_change=1.0 * np.finfo(float).eps,
        #     line_search_fn="strong_wolfe"
        # )

    def loss_func(self):

        self.optimizer.zero_grad()
        u = self.dnn(self.x_train)

        u_x = torch.autograd.grad(
            u, self.x_train,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, self.x_train,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = u_xx.squeeze(-1)
        u = u.squeeze(-1)

        loss_f = torch.mean((self.f_train + u_xx)**2)
        loss_boundary = (u[0] - self.ul)**2 + (u[-1] - self.ur)**2

        loss = loss_f + loss_boundary

        self.loss.append(loss)
        loss.backward()
        print('Iter %d, Loss: %.5e, Loss_boundary: %.5e, Loss_f: %.5e' %
              (self.iter, loss.item(), loss_boundary.item(), loss_f.item()))
        return loss

    def get_loss(self):
        return self.loss

    def train(self):
        # Construct update mesh with new coordinates
        np.random.seed(1234)
        idxs = np.arange(1, self.N-1)
        os.makedirs(self.path_model + '/Time', exist_ok=True)
        os.makedirs(self.path_model + '/Loss', exist_ok=True)

        t0 = time.time()

        while self.iter < self.max_it:
            # Take a random sample of the entire dataset
            # idxs_train = np.random.choice(
            #     idxs, size=self.batch_size, replace=False)
            # idxs_train = np.concatenate((0, idxs_train, self.N-1), axis=None)
            #idxs_valid = list(set(idxs) - set(idxs_train))

            self.x_train = self.x
            self.f_train = self.f

            #self.x_valid = self.x[idxs_valid, :]
            #self.f_valid = self.f[idxs_valid, :]

            self.dnn.train()
            self.iter += 1
            # self.optimizer.param_groups[0]['lr'] = self.lr * \
            #   10**(-np.log10(1+self.iter/100))

            # Backward and optimize
            self.optimizer.step(self.loss_func)

            if self.iter > 1:
                relErr = np.abs(self.loss[-1].item() -
                                self.loss[-2].item())/(self.loss[-2].item())
            else:
                relErr = 1

            if self.iter % 2000 == 0:
                self.dnn.save(self.path_model, self.iter)
                np.save(self.path_model +
                        f'/Time/time_{self.iter}.npy', time.time() - t0)
                np.save(self.path_model +
                        f'/Loss/Loss_{self.iter}.npy', self.loss)

            if relErr < self.relErr:
                self.dnn.save(self.path_model, self.iter)
                np.save(self.path_model +
                        f'/Time/time.npy', time.time() - t0)
                np.save(self.path_model + '/Loss.npy', self.loss)
                return

        np.save(self.path_model +
                f'/Time/time.npy', time.time() - t0)
        np.save(self.path_model + '/Loss.npy', self.loss)
        self.dnn.save(self.path_model, self.iter)
        return

    def predict(self, X):
        X = torch.tensor(X).float().unsqueeze(-1).to(self.device)
        self.dnn.eval()
        u = self.dnn(X)
        u.detach().cpu().squeeze().numpy()
        return u


def convRate(model, a, b, Nvec, device):
    Errvec = np.zeros_like(Nvec)
    rate = np.zeros_like(Nvec[:-1])

    for i, N in enumerate(Nvec):
        mesh = UnitIntervalMesh(int(N))
        x = mesh.coordinates()[:]*(b-a) + a
        mesh.coordinates()[:] = x
        X = torch.tensor(x).float().to(device)
        U = model.predict(X).detach().cpu().numpy().flatten()
        CG1 = FunctionSpace(mesh, 'CG', 1)
        dof2vertex_map = dof_to_vertex_map(CG1)
        #u_expr = Expression('pow(x[0],2.0/3.0)', degree=5)
        u_expr = Expression('sin(pi*x[0])', degree=5)
        u = Function(CG1)
        u.vector()[dof2vertex_map] = U

        err = assemble((u_expr*u_expr + u*u - 2*u_expr*u)
                       * dx(mesh), form_compiler_parameters={"quadrature_degree": 1})
        Errvec[i] = np.sqrt(err)

    rate = np.log(Errvec[:-1]/Errvec[1:])/np.log(Nvec[1:]/Nvec[:-1])
    return rate, Errvec


def convRate2(model, a, b, u_ex, Nvec):
    Errvec = np.zeros_like(Nvec)
    rate = np.zeros_like(Nvec[:-1])
    for i, N in enumerate(Nvec):
        x = np.linspace(a, b, N)
        u = model.predict(x).detach().cpu().numpy().flatten()
        f_interp = interp1d(x, u)
        err = integrate.quad(lambda x: (
            u_ex(x) - f_interp(x)) ** 2, a, b, epsabs=0)[0]
        Errvec[i] = np.sqrt(err)

    rate = np.log(Errvec[:-1]/Errvec[1:])/np.log(Nvec[1:]/Nvec[:-1])
    return rate, Errvec
