
import matplotlib.pyplot as plt
import os
import numpy as np


def conv_rate(dof, err):
    'Compute convergence rate '

    l = dof.shape[0]
    rate = np.zeros(l-1)

    for i in range(l-1):
        rate[i] = np.log(err[i]/err[i+1])/np.log(dof[i+1]/dof[i])

    return rate


full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
folder_names = ['/post', '/interp', '/href', '/uniform']

data = []

for folder_name in folder_names:
    path_data = path + folder_name
    data.append(np.load(path_data + f'/{folder_name[1:]}.npz'))

#folder_name = '/DNN/N_300_H_256_depth_15'
folder_name = '/DNN/N_300_H_20_depth_5'
path_data = path + folder_name
data.append(np.load(path_data + f'/DNN.npz'))

time = []

iter = [1000, 2000, 3000, 4000]

for i in iter:
    time.append(np.load(path_data + f'/Time/time_{i}.npy'))
loss = np.load(path_data + f'/Loss.npy')


dim_post = data[0]['dim']
L2err_post = data[0]['L2err']
hmin_post = data[0]['hmin']
time_post = data[0]['time']

dim_interp = data[1]['dim']
L2err_interp = data[1]['L2err']
hmin_interp = data[1]['hmin']
time_interp = data[1]['time']


dim_href = data[2]['dim']
L2err_href = data[2]['L2err']
hmin_href = data[2]['hmin']
time_href = data[2]['time']


dim_unif = data[3]['dim']
L2err_unif = data[3]['L2err']
hmin_unif = data[3]['hmin']
time_unif = data[3]['time']

dim_DNN = data[4]['Nvec']
L2err_DNN = data[4]['Errvec']
rate_DNN = data[4]['rate']

# plot loss
plt.figure()
plt.loglog(loss)
plt.xlabel('iteration')
plt.ylabel('loss function')

# plot time training network
plt.figure()
plt.plot(iter, time, 'k.-')
plt.xlabel('iteration')
plt.ylabel('time GPU training')

# plot difference in time
plt.figure()
plt.plot(dim_unif, time_unif, linestyle='-.', marker='o', label='uniform')
plt.plot(dim_interp, time_interp, linestyle='-.',
         marker='x', label='interpolation')
plt.plot(dim_post, time_post, linestyle='-.', marker='v', label='a-posteriori')
plt.plot(dim_href, time_href, linestyle='-.', marker='^', label='h-ref')
plt.xlabel('N')
plt.ylabel('CPU execution time')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()

rate_unif = conv_rate(dim_unif, L2err_unif)
rate_interp = conv_rate(dim_interp, L2err_interp)
rate_post = conv_rate(dim_post, L2err_post)
rate_href = conv_rate(dim_href, L2err_href)

plt.figure()
plt.plot(dim_unif, L2err_unif, linestyle='-.', marker='o',
         label='uniform | rate %4g' % rate_unif[-1])
plt.plot(dim_DNN, L2err_DNN, linestyle='-.', marker='D',
         label='DNN | rate %4g' % rate_DNN[-1])
plt.plot(dim_interp, L2err_interp, linestyle='-.', marker='x',
         label='interpolation | rate %4g' % rate_interp[-1])
plt.plot(dim_post, L2err_post, linestyle='-.', marker='v',
         label='a-posteriori | rate %4g' % np.mean(rate_post))
plt.plot(dim_href, L2err_href, linestyle='-.', marker='^',
         label='h-ref | rate %4g' % np.mean(rate_href[-10:]))
plt.xlabel('N')
plt.ylabel('L2-error')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
