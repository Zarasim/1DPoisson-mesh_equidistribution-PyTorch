import torch
from torch import nn
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from collections import OrderedDict
from torch._C import device
import torch.nn.functional as F
import time
import os
import torch.autograd as grad


def L2error(a, b, f, g, degree=5):
    return np.sqrt(integrate.fixed_quad(lambda x: (f(x) - g(x))**2, a, b, n=degree)[0])


class LinearNeuralTangentKernel(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, beta=0.1, w_sig=1):
        self.beta = beta

        # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        super().__init__(in_features, out_features)
        self.reset_parameters()
        self.w_sig = w_sig

    def reset_parameters(self):
        # Fills the tensor with values drawn from a normal distribution
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, input):
        # (input, weight, bias=None) Applies a linear transformation to the incoming data: y=xAT+b
        return F.linear(input, self.w_sig * self.weight/np.sqrt(self.in_features), self.beta * self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, beta={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.beta
        )


class WideNet(nn.Module):

    def __init__(self, n_wid=100, n_in=1, n_out=1, beta=1.0):
        super(WideNet, self).__init__()
        self.fc1 = LinearNeuralTangentKernel(n_in, n_wid, beta=beta)
        self.fc2 = LinearNeuralTangentKernel(n_wid, n_out, beta=beta)
        self.sigma = torch.nn.Tanh()

    def forward(self, x):
        x = self.sigma(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        x = self.sigma(self.fc1(x))
        x = self.fc2(x)
        x = x.detach().cpu().squeeze().numpy().flatten()
        return x

    def save(self, path_model, iter):
        # Python dictionary object that maps each layer to its parameter tensor.
        torch.save(self.state_dict(), path_model + f'/DNN/DNN_{iter}.pth')

    def load(self, path):
        # load the parameters from pth file
        self.load_state_dict(torch.load(path))


class PINN():

    def __init__(self, x_bc, y_bc, x_r, y_r, nwidth, device, path_model, lambda_b, lr=1e-5, net=[]):

        self.mu_x, self.sigma_x = x_r.mean(), x_r.std()

        # Normalize domain variables
        self.x_bc = x_bc.to(device)
        self.x_r = x_r.to(device)

        self.y_bc = y_bc.to(device)
        self.y_r = y_r.to(device)

        self.nwidth = nwidth
        self.device = device

        self.lambda_r = 1.9851
        self.lambda_b = lambda_b

        # neural net
        self.loss = []
        self.lr = lr

        if net == []:
            print('net created')
            self.dnn = WideNet(n_wid=self.nwidth).to(self.device)
            # manual_seed works only with CUDA
            torch.manual_seed(1234)
        else:
            print('net loaded')
            self.dnn = net

        self.y_bc_pred = self.dnn(self.x_bc)
        self.y_r_pred = self.dnn(self.x_r)
        self.path_model = path_model
        self.iter = 0

        # optimizers
        self.optimizer = torch.optim.SGD(self.dnn.parameters(), lr=self.lr)

    def loss_func(self):

        self.optimizer.zero_grad()
        u = self.dnn(self.x_r)

        u_x = torch.autograd.grad(
            u, self.x_r,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, self.x_r,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        u = self.dnn(self.x_bc)

        self.loss_f = self.lambda_r*torch.mean((self.y_r - u_xx)**2)
        self.loss_bc = self.lambda_b*torch.mean((self.y_bc - u)**2)

        loss = self.loss_f + self.loss_bc
        loss.backward(retain_graph=True)

        if self.iter % (self.max_iter//20) == 0:
            self.loss.append(loss.item())
        print('Iter %d, Loss: %.5e, Loss_boundary: %.5e, Loss_f: %.5e' %
              (self.iter, loss.item(), self.loss_bc.item(), self.loss_f.item()))
        return loss

    def get_loss(self):
        return self.loss

    def get_net(self):
        return self.dnn

    def train(self, x_boundary, N, u_ex, max_iter=10000):
        # Construct update mesh with new coordinates
        self.max_iter = max_iter
        np.random.seed(1234)
        self.iter = 0
        x = np.linspace(x_boundary[0], x_boundary[1], N)
        interps_ex = u_ex, u_ex, u_ex

        os.makedirs(self.path_model + '/Var', exist_ok=True)
        os.makedirs(self.path_model + '/DNN', exist_ok=True)
        t0 = time.time()
        while self.iter < max_iter:

            self.dnn.train()
            self.iter += 1
            # the bigger the denominator, the slower the decay
            self.optimizer.param_groups[0]['lr'] = self.lr * \
                np.exp(-self.iter/(5*10**3))

            # Backward and optimize
            self.optimizer.step(self.loss_func)

            if self.iter % (max_iter//20) == 0:
                self.dnn.save(self.path_model, self.iter)
                u = self.dnn.predict(torch.tensor(
                    x, dtype=torch.float32).unsqueeze(-1).to(self.device))
                interps = interp1d(x, u, kind='linear'), interp1d(
                    x, u, kind='quadratic'), interp1d(x, u, kind='cubic')
                errs = [L2error(x_boundary[0], x_boundary[1], f, g, degree=i+1)
                        for i, (f, g) in enumerate(zip(interps_ex, interps))]
                np.savez(self.path_model + f'/Var/vars_{self.iter}.npz', time=time.time() - t0, loss_b=self.loss_bc.item(), loss_res=self.loss_f.item(),
                         errs=errs, lr=self.optimizer.param_groups[0]['lr'])

        return
