from PINN_Mesh import DNN_Mesh, PINN_Mesh
from PINN_Poisson import DNN_Poisson, PINN_Poisson, convRate

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# CUDA support

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f"The device used to train the net is {device}")

# create uniform mesh with N points
N = 100
a = -1.0
b = 1.0
xi = np.linspace(a, b, N)


def u_ex(x): return np.sin(np.pi*x)


def f(x): return (np.pi**2)*np.sin(np.pi*x)


# boundary conditions
ul = 0.0
ur = 0.0


## configure path ##
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)


# configure parameters of Network Mesh and Network Poisson #
H = [20, 20]
hidden_depth = [5, 2]

model_name_mesh = f'/DNN_mesh/N_{N}_H_{H[0]}_depth_{hidden_depth[0]}'
path_model_Mesh = path + model_name_mesh

if not(os.path.exists(path_model_Mesh)):
    os.makedirs(path_model_Mesh)

model_name_Poisson = f'/DNN_Poisson/N_{N}_H_{H[1]}_depth_{hidden_depth[1]}'
path_model_Poisson = path + model_name_Poisson

if not(os.path.exists(path_model_Poisson)):
    os.makedirs(path_model_Poisson)


layers_mesh = [1] + [H[0]]*hidden_depth[0] + [1]
layers_Poisson = [1] + [H[1]]*hidden_depth[1] + [1]


MeshNet = []
PoissonNet = []

ep = 1
epochs = 2

X = xi
PoissonModel = PINN_Poisson(X, f, layers_Poisson, ul, ur, device, path_model_Poisson,
                            max_it=1000, lr=1e-3, relErr=1e-8, net=PoissonNet)
PoissonModel.train(ep, X, f)
u = PoissonModel.predict(X)

MeshModel = PINN_Mesh(N, xi, u, layers_mesh, a, b, device, path_model_Mesh,
                      max_it=20, lr=1e-3, relErr=1e-8, net=MeshNet)


for i in range(1, epochs+1):

    MeshModel.train(ep, X, u)
    print('Mesh trained')
    X = MeshModel.predict(xi)

    ep += i
    PoissonModel.train(ep, X, f)
    print('Poisson trained')
    u = PoissonModel.predict(X)

# define n_epochs and train for that epochs

LossMesh = MeshModel.get_loss()
LossPoisson = PoissonModel.get_loss()

plt.figure(figsize=(10, 5), dpi=80)
plt.plot(LossMesh)
plt.show()


plt.figure(figsize=(10, 5), dpi=80)
plt.plot(LossPoisson)
plt.show()


# print function
plt.figure(figsize=(10, 5), dpi=80)
plt.scatter(X, u, s=1)
plt.plot(X, u_ex(X))
plt.show()


Nvec = np.logspace(1.0, 3.7, 7)
rate, Errvec = convRate(PoissonModel, a, b, Nvec, device)

# print conv rate
#plt.plot(Nvec, Errvec_unif, 'g-')
plt.plot(Nvec, Errvec, 'r-.')
plt.plot(Nvec, Nvec**(-2))
plt.xlabel('N')
plt.ylabel('L2 error')
plt.yscale('log')
plt.xscale('log')
plt.show()


#print(f"conv rate adapted mesh is {rate}")
#np.savez(path_model + '/DNN.npz', Errvec=Errvec, Nvec=Nvec, rate=rate)
#np.save(path_model + '/Loss.npy', Loss)
