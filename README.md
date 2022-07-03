# Poisson-1D_mesh_equidistribution_Pytorch

his repository employs a neural network to solve the Poisson's equation in 1D and equidistribute the mesh points to increase the accuracy.

The PINN_Mesh.py file defines the neural network for equidistributing the mesh points.

The PINN_Poisson.py file defins the neural network for solving the Poisson's equation in 1D.

The trainDNN.py is the main file that trains both networks alternatively to yields the solution.

The trained network can be found in the folder DNN_Poisson.

The library used for this project is PyTorch.
