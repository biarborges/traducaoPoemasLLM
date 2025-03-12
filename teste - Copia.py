import torch
print(torch.cuda.is_available())  # Só retorna True para NVIDIA
print(torch.backends.opencl.is_available())  # OpenCL (AMD/Intel)
