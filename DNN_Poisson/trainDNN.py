from PINNPoisson import *
import matplotlib.pyplot as plt
import os

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"The device used to train the net is {device}")

# create uniform mesh with N points
N = 500
ul = 0.0
ur = 0.0


#def u_ex(x): return x**(2.0/3.0)
#def f(x): return -(2./9.)*(x+1e-8)**(-4./3.)


def u_ex(x): return np.sin(np.pi*x)


def f(x): return (np.pi**2)*np.sin(np.pi*x)


a = -1
b = 1
X = np.linspace(a, b, N)

## Training ##

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

H = 20
hidden_depth = 5
model_name = f'/Nets/N_{N}_H_{H}_depth_{hidden_depth}'
path_model = path + model_name

if not(os.path.exists(path_model)):
    os.makedirs(path_model)

# model parameters
layers = [1] + [H]*hidden_depth + [1]


net = DNN(layers)
net.load(path_model + f'/DNN_{2000}.pth')
net.to(device)
net.eval()

#net = []
model = PINN(X, f, layers, ul, ur, device, path_model, batch_size=498,
             max_it=50000, lr=5e-7, relErr=1e-12, net=net)

# define n_epochs and train for that epochs
# model.train()
#loss = model.get_loss()
loss = np.load(path_model + '/Loss.npy', allow_pickle=True)
plt.figure(figsize=(10, 5), dpi=80)
plt.loglog(loss)
plt.show()


U = model.predict(X).detach().cpu().numpy().flatten()

# print function
plt.figure(figsize=(10, 5), dpi=80)
plt.scatter(X, u_ex(X), s=5, label='u_exact')
plt.scatter(X, U, s=5, label='u_PINN')
plt.xlabel('Computational coordinates')
plt.ylabel('u')
plt.legend()
plt.show()


Nvec = np.logspace(1.0, 3.7, 7)
#rate, Errvec = convRate2(model, a, b, u_ex, Nvec)
rate, Errvec = convRate(model, a, b, Nvec, device)

# print conv rate
plt.plot(Nvec, Errvec, 'g.-')
plt.plot(Nvec, Nvec**(-2), alpha=0.3)
plt.xlabel('N')
plt.ylabel('L2 error')
plt.yscale('log')
plt.xscale('log')
plt.show()


print(f"conv rate adapted mesh is {rate}")
# np.savez(path_model + '/DNN.npz', Errvec=Errvec, Nvec=Nvec, rate=rate)
# np.save(path_model + '/Loss.npy', loss)
