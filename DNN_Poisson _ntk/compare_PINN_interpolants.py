import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from PINNPoisson import DNN, L2error
from scipy.interpolate import interp1d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

N = 500
H = 50
Depth = 2

layers = [1] + [H]*Depth + [1]

var_name = f'/Nets/N_{N}_H_{H}_depth_{Depth}/Var'
model_name = f'/Nets/N_{N}_H_{H}_depth_{Depth}/DNN'

path_var = path + var_name
path_model = path + model_name


max_iter = 50000
step = max_iter//20
net = DNN(layers)
net.load(path_model + f'/DNN_{max_iter}.pth')
net.to(device)
net.eval()

a = -1.0
b = 1.0
x = np.linspace(a, b, N)
u = net.predict(x, device)


def u_ex(x): return np.sin(np.pi*x)


# print function
plt.figure(figsize=(10, 5), dpi=80)
plt.plot(x, u_ex(x), linewidth=1, label='u_exact')
plt.scatter(x, u, s=5, c='red', label='u_PINN')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()

Nvec = np.logspace(1, 3, 10)
L2err = []
for Nv in Nvec:
    x = np.linspace(a, b, Nv)
    u = net.predict(x, device)
    u_interp = interp1d(x, u, kind='linear')
    L2err.append(L2error(a, b, u_ex, u_interp, degree=7))


rate = np.log(L2err[-1]/L2err[-2])/np.log(Nvec[-2]/Nvec[-1])

# print conv rate
plt.plot(Nvec, L2err, 'g.-', label=f'rate: {round(rate,2)}')
plt.plot(Nvec, Nvec**(-2), alpha=0.3, label='rate: 2')
plt.xlabel('N')
plt.ylabel('L2 error')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.title(f'Network_N_{N}_H_{H}_depth_{Depth}_it_{max_iter}')
plt.show()


##############################################

iters = list(range(step, max_iter, step))

data = []
for iter in iters:
    data.append(np.load(path_var +
                f'/vars_{int(iter)}.npz', allow_pickle=True))


time = [data[i]['time'] for i in range(len(iters))]
loss = [data[i]['loss'] for i in range(len(iters))]
errs = np.array([data[i]['errs'] for i in range(len(iters))])
lr = [data[i]['lr'] for i in range(len(iters))]

err1 = errs[:, 0]
err2 = errs[:, 1]
err3 = errs[:, 2]


plt.figure()
plt.plot(iters, time, '.-')
plt.xlabel('iterations')
plt.ylabel('training time (GPU) [s]')
plt.title(
    f'Network_N_{N}_H_{H}_depth_{Depth} trained for increasing iterations')
plt.show()


plt.figure()
plt.plot(iters, loss, 'k.-', label='loss')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.title(
    f'Network_N_{N}_H_{H}_depth_{Depth}')
plt.legend()
plt.show()


plt.figure()
plt.plot(iters, loss, label='loss')
plt.plot(iters, err1, '-', label='err1')
plt.plot(iters, err2, '--', label='err2')
plt.plot(iters, err3, '.-', label='err3')
plt.xlabel('iterations')
plt.ylabel('Error')
plt.xscale('log')
plt.yscale('log')
plt.title(
    f'Network_N_{N}_H_{H}_depth_{Depth}')
plt.legend()
plt.show()


plt.figure()
plt.plot(iters, lr, 'k.-')
plt.xlabel('iterations')
plt.ylabel('learning rate')
plt.title(
    f'Network_N_{N}_H_{H}_depth_{Depth}')
plt.xscale('log')
plt.yscale('log')
plt.show()
