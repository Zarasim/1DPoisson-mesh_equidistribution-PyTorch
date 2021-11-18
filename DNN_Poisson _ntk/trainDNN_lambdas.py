from PINNPoisson import *
from fenics import *
import matplotlib.pyplot as plt
import os

# CUDA support
dev = torch.device('cuda'if torch.cuda.is_available() else 'cpu')


print(f"The device used to train the net is {dev}")

a = 4


def g(x, a): return torch.tensor(0)


def u(x, a): return torch.sin(np.pi * a * x)


def u_xx(x, a): return -((np.pi*a)**2)*torch.sin(np.pi * a * x)

# model parameters


H = 100
Depth = 1
layers = [1] + [H]*Depth + [1]

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

lambdas = np.logspace(0, 3, 20)

N_r = 100
N_b = 100
x_boundary = np.array([-1.0, 1.0])
x_r = torch.linspace(x_boundary[0], x_boundary[1], N_r,
                     requires_grad=True, dtype=torch.float32).unsqueeze(-1)
x_bc1 = x_boundary[0]*torch.ones((N_b // 2, 1),
                                 requires_grad=True, dtype=torch.float32)
x_bc2 = x_boundary[1]*torch.ones((N_b // 2, 1),
                                 requires_grad=True, dtype=torch.float32)
x_bc = torch.vstack((x_bc1, x_bc2))

y_r = u_xx(x_r, a)
y_bc = g(x_bc, a)


def u_ex(x): return np.sin(np.pi * a * x)


for lambda_b in lambdas:

    ## Training ##
    model_name = f'/Nets_lambdas/Nr_{N_r}_H_{H}_depth_{Depth}_lambdabc_{lambda_b}'
    path_model = path + model_name
    os.makedirs(path_model, exist_ok=True)

    model = PINN(x_bc, y_bc, x_r, y_r, H, dev,
                 path_model, lambda_b, lr=1e-3)

    # define n_epochs and train for that epochs
    model.train(x_boundary, N_r, u_ex, max_iter=50000)
