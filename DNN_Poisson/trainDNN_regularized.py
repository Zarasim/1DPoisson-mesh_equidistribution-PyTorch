from PINNPoisson import *
from fenics import *
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


a = -1
b = 1

# model parameters
H = 50
hidden_depth = 2
layers = [1] + [H]*hidden_depth + [1]


full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)


Nvec = np.logspace(1, 4, 10)
Errvec = []

for N in Nvec:

    X = np.linspace(a, b, N)

    ## Training ##

    model_name = f'/Nets_Regularized/N_{N}_H_{H}_depth_{hidden_depth}'
    path_model = path + model_name

    if not(os.path.exists(path_model)):
        os.makedirs(path_model)

    # net = DNN(layers)
    # net.load(path_model + f'/DNN/DNN_{80000}.pth')
    # net.to(device)
    # net.eval()

    net = []
    model = PINN(X, f, layers, ul, ur, device, path_model,
                 max_it=25000, lr=1e-3, relErr=1e-8, net=net)

    # define n_epochs and train for that epochs
    model.train()

    mesh = UnitIntervalMesh(int(N))
    x = mesh.coordinates()[:]*(b-a) + a
    mesh.coordinates()[:] = x
    X = torch.tensor(x).float().to(device)
    U = model.predict(X).detach().cpu().numpy().flatten()
    CG1 = FunctionSpace(mesh, 'CG', 1)
    dof2vertex_map = dof_to_vertex_map(CG1)
    u_expr = Expression('sin(pi*x[0])', degree=5)
    u = Function(CG1)
    u.vector()[dof2vertex_map] = U
    err = assemble((u_expr*u_expr + u*u - 2*u_expr*u)
                   * dx(mesh), form_compiler_parameters={"quadrature_degree": 1})
    Errvec.append(np.sqrt(err))


# print conv rate
plt.plot(Nvec, Errvec, 'g.-')
plt.plot(Nvec, Nvec**(-2), alpha=0.3, label='rate: 2')
plt.xlabel('Number of training data')
plt.ylabel('L2 error')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
