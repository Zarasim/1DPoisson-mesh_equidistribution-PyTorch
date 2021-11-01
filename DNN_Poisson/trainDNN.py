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

ul = 0.0
ur = 0.0

#def u_ex(x): return x**(2.0/3.0)


def u_ex(x): return np.sin(np.pi*x)


#def f(x): return -(2./9.)*(x+1e-8)**(-4./3.)
def f(x): return (np.pi**2)*np.sin(np.pi*x)


N = 500
a = -1
b = 1
X = np.linspace(a, b, N)

## Training ##

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

H = 50
hidden_depth = 2
model_name = f'/Nets/N_{N}_H_{H}_depth_{hidden_depth}'
path_model = path + model_name

if not(os.path.exists(path_model)):
    os.makedirs(path_model)

# model parameters
layers = [1] + [H]*hidden_depth + [1]


net = DNN(layers)
net.load(path_model + f'/DNN/DNN_{80000}.pth')
net.to(device)
net.eval()

#net = []
model = PINN(X, f, layers, ul, ur, device, path_model,
             max_it=80000, lr=1e-3, relErr=1e-8, net=net)

# define n_epochs and train for that epochs
# model.train()


X = np.linspace(a, b, 10000)
U = model.predict(X).detach().cpu().numpy().flatten()

# print function
plt.figure(figsize=(10, 5), dpi=80)
plt.plot(X, u_ex(X), linewidth=2, label='u_exact')
plt.scatter(X, U, s=5, c='red', label='u_PINN')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()


Nvec = np.logspace(1.0, 3.7, 7)
rate, Errvec = convRate(model, a, b, Nvec, device)

# print conv rate
plt.plot(Nvec, Errvec, 'g.-')
plt.plot(Nvec, Nvec**(-2), alpha=0.3, label='rate: 2')
plt.xlabel('N')
plt.ylabel('L2 error')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
print(f"conv rate adapted mesh is {rate}")
