from PINN import *
import matplotlib.pyplot as plt
import os

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"The device used to train the net is {device}")

# create uniform mesh with N points
N = 300
a = 0
b = 1.0


def u(x): return x**(2.0/3.0)


xi = np.linspace(a, b, N)

## Training ##


full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
max_it = 50000
H = 256
hidden_depth = 15
model_name = f'/DNN/N_{N}_H_{H}_depth_{hidden_depth}'
path_model = path + model_name

if not(os.path.exists(path_model)):
    os.makedirs(path_model)

# model parameters
layers = [1] + [H]*hidden_depth + [1]


iter = 9000
net = DNN(layers, a, b)
net.load(path_model + f'/DNN_{iter}.pth')
net.to(device)
net.eval()

# # Set default weights of the network
# dic = net.state_dict()
# for k in dic:
#     dic[k] *= 0
# net.load_state_dict(dic)
# del(dic)
#net = []
model = PhysicsInformedNN(N, xi, u, layers, a, b,
                          device, path_model, max_it, lr=1e-4, relErr=1e-8, net=net)

# define n_epochs and train for that epochs
# model.train()
#Loss = model.get_loss()
# plt.figure(figsize=(10, 5), dpi=80)
# plt.plot(Loss[100:])
# plt.show()

X = model.predict(xi).detach().cpu().numpy()

# print function
plt.figure(figsize=(10, 5), dpi=80)
plt.scatter(X, u(X), s=1)
plt.xlabel('Computational coordinates')
plt.ylabel('u')
plt.show()


Nvec = np.logspace(1.0, 3.7, 7)
rate_unif, Errvec_unif = convRate_fenics(model, a, b, Nvec, u, equid=True)
rate, Errvec = convRate_fenics(model, a, b, Nvec, u, equid=False)

# print conv rate
plt.plot(Nvec, Errvec_unif, 'g.-', label='uniform')
plt.plot(Nvec, Errvec, 'r.-', label='adapted')
plt.plot(Nvec, Nvec**(-2), alpha=0.3)
plt.xlabel('N')
plt.ylabel('L2 error')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()


print(f"conv rate adapted mesh is {rate}")

#np.savez(path_model + '/DNN.npz', Errvec=Errvec, Nvec=Nvec, rate=rate)
#np.save(path_model + '/Loss.npy', Loss)
