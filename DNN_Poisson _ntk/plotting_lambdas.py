import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.core.fromnumeric import reshape


cmap = plt.cm.get_cmap('gist_ncar')
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

N = 100
H = 100
Depth = 1

lambdas = np.logspace(0, 3, 20)[:12]
iters = np.linspace(2500, 50000, 20)

time = dict()
loss_b = dict()
loss_res = dict()
loss = dict()
err = dict()
lr = dict()

for lambda_ in lambdas:
    data = []
    folder_name = f'/Nets_lambdas/Nr_{N}_H_{H}_depth_{Depth}_lambdabc_{lambda_}/Var'

    for iter in iters:
        data.append(np.load(path + folder_name +
                            f'/vars_{int(iter)}.npz', allow_pickle=True))

    time[lambda_] = [data[i]['time'] for i in range(len(iters))]
    loss_b[lambda_] = [data[i]['loss_b'] for i in range(len(iters))]
    loss_res[lambda_] = [data[i]['loss_res'] for i in range(len(iters))]
    loss[lambda_] = [loss_b[lambda_][i] + loss_res[lambda_][i]
                     for i in range(len(iters))]

    err[lambda_] = np.array([data[i]['errs']
                            for i in range(len(iters))]).reshape(-1, 3)
    lr[lambda_] = [data[i]['lr'] for i in range(len(iters))]


plt.figure()
for i, lambda_ in enumerate(lambdas):
    plt.plot(iters, loss_b[lambda_], '.-', c=cmap(i/19), linewidth=3 - 2*(i)/19,
             label=f'lambda: {round(lambda_,2)}')
plt.xlabel('iterations')
plt.ylabel('Loss_boundary')
plt.xscale('log')
plt.yscale('log')
plt.title(
    f'Network_N_{N}_H_{H}_depth_{Depth} trained for increasing iterations')
plt.legend()
plt.show()


plt.figure()
for i, lambda_ in enumerate(lambdas):
    plt.plot(iters, loss_res[lambda_], '.-', c=cmap(i/19), linewidth=3 - 2*(i)/19,
             label=f'lambda: {round(lambda_,2)}')
plt.xlabel('iterations')
plt.ylabel('Loss_residual')
plt.xscale('log')
plt.yscale('log')
plt.title(
    f'Network_N_{N}_H_{H}_depth_{Depth}')
plt.legend()
plt.show()


plt.figure()
for i, lambda_ in enumerate(lambdas):
    plt.plot(iters, loss[lambda_], '.-', c=cmap(i/19), linewidth=3 - 2*(i)/19,
             label=f'lambda: {round(lambda_,2)}')
plt.xlabel('iterations')
plt.ylabel('Total Loss')
plt.xscale('log')
plt.yscale('log')
plt.title(
    f'Network_Nr_{N}_H_{H}_depth_{Depth}')
plt.legend()
plt.show()


plt.figure()
for i, lambda_ in enumerate(lambdas):
    plt.plot(iters, err[lambda_][:, 0], '.-', c=cmap(i/19), linewidth=3 - 2*(i)/19,
             label=f'lambda: {round(lambda_,2)}')
plt.xlabel('iterations')
plt.ylabel('L2 error')
plt.xscale('log')
plt.yscale('log')
plt.title(
    f'Network_Nr_{N}_H_{H}_depth_{Depth} 1st order interpolant')
plt.legend()
plt.show()


plt.figure()
for i, lambda_ in enumerate(lambdas):
    plt.plot(iters, err[lambda_][:, 1], '.-', c=cmap(i/19), linewidth=3 - 2*(i)/19,
             label=f'lambda: {round(lambda_,2)}')
plt.xlabel('iterations')
plt.ylabel('L2 error')
plt.xscale('log')
plt.yscale('log')
plt.title(
    f'Network_Nr_{N}_H_{H}_depth_{Depth} 2nd order interpolant')
plt.legend()
plt.show()


plt.figure()
for i, lambda_ in enumerate(lambdas):
    plt.plot(iters, err[lambda_][:, 2], '.-', c=cmap(i/19), linewidth=3 - 2*(i)/19,
             label=f'lambda: {round(lambda_,2)}')
plt.xlabel('iterations')
plt.ylabel('L2 error')
plt.xscale('log')
plt.yscale('log')
plt.title(
    f'Network_Nr_{N}_H_{H}_depth_{Depth} 3rd order interpolant')
plt.legend()
plt.show()


plt.figure()
plt.plot(iters, lr[lambdas[0]], 'k.-')
plt.xlabel('iterations')
plt.ylabel('learning rate')
plt.title(
    f'Network_N_{N}_H_{H}_depth_{Depth} trained for increasing iterations')
plt.xscale('log')
plt.yscale('log')
plt.show()


plt.figure()
plt.plot(iters, time[lambdas[0]], '.-')
plt.xlabel('iterations')
plt.ylabel('training time (GPU) [s]')
plt.title(
    f'Network_N_{N}_H_{H}_depth_{Depth} trained for increasing iterations')
plt.show()
