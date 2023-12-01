import random

import torch
import numpy as np
import os
import glob

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from qqdm import qqdm


def same_seeds(seed):
  # Python built-in random module
  random.seed(seed)
  # Numpy
  np.random.seed(seed)
  # Torch
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

# Set random seed for reproducibility
same_seeds(323)




workspace_dir = '.'
# Root directory for dataset
dataroot = "anime_face_dataset"
# Number of workers for dataloader
workers = 24
# Batch size during training
batch_size = 768
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 10
# Size of feature maps in generator
ngf = 256
# Size of feature maps in discriminator
ndf = 256
# Number of training epochs
num_epochs = 400
# Learning rate for optimizers
lr = 0.0001
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


class Generator(nn.Module):
  """
  Input shape: (N, in_dim, 1, 1)
  Output shape: (N, nc, image_size, image_size)

  In our sample code, input/output shape are:
    Input shape: (N, 100, 1, 1)
    Output shape: (N, 3, 64, 64)
  """

  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      # Input is Z, going into a convolution
      nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 8),
      nn.ReLU(True),
      # State size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # State size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # State size. (ngf*2) x 16 x 16
      nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # State size. (ngf) x 32 x 32
      nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
      nn.Tanh()
      # State size. (nc) x 64 x 64
    )

  def forward(self, input):
    return self.main(input)

class Discriminator(nn.Module):
  """
  Input shape: (N, nc, image_size, image_size)
  Output shape: (N, )

  In our sample code, input/output are:
    Input shape: (N, 3, 64, 64)
    Output shape: (N, )
  """
  def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      # Input is (nc) x 64 x 64
      nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # State size. (ndf) x 32 x 32
      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # State size. (ndf*2) x 16 x 16
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # State size. (ndf*4) x 8 x 8
      nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # State size. (ndf*8) x 4 x 4
      nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
      nn.Sigmoid()
    )

  def forward(self, input):
    return self.main(input)
  


# Create the generator
netG = Generator(ngpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
  netG = nn.DataParallel(netG, list(range(ngpu)))
# Print the model
print(netG)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
  netD = nn.DataParallel(netD, list(range(ngpu)))
# Print the model
print(netD)



# load gernerator
ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
device = torch.device("cuda")
netG = Generator(ngpu)
netG.load_state_dict(torch.load(os.path.join(ckpt_dir, 'netG.pth')))
netG.eval()
netG.to(device)




##########################################################################
# TODO: Store your generate images in 5*5 Grid
# Implementation A.1-3
##########################################################################
# Generate 100 images and make a grid to save them.
n_output = 100
fixed_noise = torch.randn(25, nz, 1, 1, device=device)
imgs_sample = (netG(fixed_noise).data + 1) / 2.0

grid_img = torchvision.utils.make_grid(imgs_sample[:25].cpu(), nrow=5)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.savefig('result.png', bbox_inches='tight')




##########################################################################
# TODO: Interpolation 3 pairs of z vectors and plot a 3*10 image.
# Implementation A.1-4
##########################################################################
fixed_noise_1 = torch.randn(3, nz, 1, 1, device=device)
fixed_noise_2 = torch.randn(3, nz, 1, 1, device=device)

interpolation = fixed_noise_1.clone()
for i in range(1, 10):
    v = (1.0 - 0.1*i) * fixed_noise_1 + 0.1*i * fixed_noise_2
    interpolation = torch.cat([interpolation,v])
# print(interpolation.reshape(3,10,nz,1,1).permute(1,0,2,3,4).reshape(-1,nz,1,1))
interpolation = interpolation.reshape(10,3,nz,1,1).permute(1,0,2,3,4).reshape(-1,nz,1,1)

imgs_sample = (netG(interpolation).data + 1) / 2.0

grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.savefig('interpolation.png', bbox_inches='tight')