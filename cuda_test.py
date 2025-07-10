import torch

# Determine if an NVIDIA GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))