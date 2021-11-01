import torch
from torch import nn
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from dolfin import *
from collections import OrderedDict
import time
import os


# The deep neural network
class DNN_Poisson(torch.nn.Module):

    # layers count the hiddens + the output
    def __init__(self, layers):

        super().__init__()

       # parameters
        self.depth = len(layers) - 1
        #self.dropout = torch.nn.Dropout(0.1)
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
        torch.save(self.state_dict(), path_model + f'/DNN/DNN_{iter}.pth')

    def load(self, path):
        # load the parameters from pth file
        self.load_state_dict(torch.load(path))


class PINN_Poisson():

    def __init__(self, X, f, layers, ul, ur, device, path_model, max_it=1000, lr=0.1, relErr=1e-8, net=[]):

        self.device = device
        self.ul = torch.tensor(ul).float().to(self.device)
        self.ur = torch.tensor(ur).float().to(self.device)
        self.x = torch.tensor(X, requires_grad=True).float().to(
            self.device).unsqueeze(-1)

        self.N = len(X)
        self.layers = layers
        self.f = torch.tensor(f(X)).to(self.device)
        self.relErr = relErr

        # neural net
        self.loss = []
        self.lr = lr

        if net == []:
            print('net created')
            self.dnn = DNN_Poisson(self.layers).to(self.device)
            # manual_seed works only with CUDA
            torch.manual_seed(1234)
            self.dnn.apply(init_weights)
        else:
            print('net loaded')
            self.dnn = net

        self.max_it = max_it
        self.path_model = path_model
        self.iter = 0
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=self.lr)

    def loss_func(self):

        self.optimizer.zero_grad()
        u = self.dnn(self.x)

        u_x = torch.autograd.grad(
            u, self.x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, self.x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        u = u.squeeze(-1)
        u_xx = u_xx.squeeze(-1)

        loss_f = torch.mean((self.f + u_xx)**2)
        loss_boundary = 0.5*(u[0] - self.ul)**2 + (u[-1] - self.ur)**2

        loss = loss_f + loss_boundary
        loss.backward()

        if self.iter % 2000 == 0:
            self.loss.append(loss.item())

        print('Iter %d, Loss: %.5e, Loss_boundary: %.5e, Loss_f: %.5e' %
              (self.iter, loss.item(), loss_boundary.item(), loss_f.item()))
        return loss

    def get_loss(self):
        return self.loss

    def train(self, ep, X, f):

        # Construct update mesh with new coordinates
        np.random.seed(1234)
        self.x = torch.tensor(X, requires_grad=True).float().to(
            self.device).unsqueeze(-1)
        self.f = torch.tensor(f(X)).to(self.device)

        os.makedirs(self.path_model + '/Var', exist_ok=True)
        os.makedirs(self.path_model + '/DNN', exist_ok=True)
        while self.iter < self.max_it*ep:

            self.dnn.train()
            self.iter += 1
            self.optimizer.param_groups[0]['lr'] = self.lr * \
                np.exp(-self.iter/(5*10**3))

            self.optimizer.step(self.loss_func)

            if self.iter % (self.max_it//2) == 0:
                self.dnn.save(self.path_model, self.iter)
                np.savez(self.path_model + f'/Var/vars_{self.iter}.npz', loss=self.loss,
                         lr=self.optimizer.param_groups[0]['lr'])
                relErr = np.abs(self.loss[-1]-self.loss[-2]) / \
                    self.loss[-2] if len(self.loss) > 1 else 1
                # if relErr < self.relErr:
                #     return

        np.savez(self.path_model + f'/Var/vars_{self.iter}.npz',
                 loss=self.loss, lr=self.optimizer.param_groups[0]['lr'])
        self.dnn.save(self.path_model, self.iter)
        return

    def predict(self, X):
        X = torch.tensor(X).float().unsqueeze(-1).to(self.device)
        self.dnn.eval()
        u = self.dnn(X)
        u = u.detach().cpu().squeeze().numpy()
        return u


def init_weights(m):
    if type(m) == torch.nn.Linear:
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.zero_()


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
