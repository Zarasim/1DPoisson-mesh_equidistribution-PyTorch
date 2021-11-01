import matplotlib.pyplot as plt
import os
import numpy as np


full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

N = 500
H = 50
Depth = 2

folder_name = f'/Nets/N_{N}_H_{H}_depth_{Depth}/Var'

iters = list(range(2000, 64000, 2000))


data = []
for iter in iters:
    data.append(np.load(path + folder_name +
                f'/vars_{int(iter)}.npz', allow_pickle=True))


time = [data[i]['time'] for i in range(len(iters))]
loss = [data[i]['loss'] for i in range(len(iters))]
err = [data[i]['err'] for i in range(len(iters))]
lr = [data[i]['lr'] for i in range(len(iters))]

plt.figure()
plt.plot(iters, time, '.-')
plt.xlabel('iterations')
plt.ylabel('training time (GPU) [s]')
plt.show()


plt.figure()
plt.plot(iters, loss[-1], 'k.-', label='loss')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()


plt.figure()
plt.plot(iters, loss[-1], 'k.-', label='loss')
plt.plot(iters, err, 'g.-', label='err')
plt.xlabel('iterations')
plt.ylabel('Error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

plt.figure()
plt.plot(iters, lr, 'k.-')
plt.xlabel('iterations')
plt.ylabel('learning rate')
plt.xscale('log')
plt.yscale('log')
plt.show()
