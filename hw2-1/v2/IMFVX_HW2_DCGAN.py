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
same_seeds(999)




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
nz = 16
# Size of feature maps in generator
ngf = 256
# Size of feature maps in discriminator
ndf = 128
# Number of training epochs
num_epochs = 500
# Learning rate for optimizers
lr = 0.0001
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# log(img) and checkpoints directory
log_dir = os.path.join(workspace_dir, 'logs')
ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)




def get_dataset(dataroot):
  dataset = dset.ImageFolder(root=dataroot,
                transform=transforms.Compose([
                  transforms.Resize(image_size),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
  return dataset

# Create the dataset
dataset = get_dataset(os.path.join(workspace_dir, dataroot))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)


# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig('TrainingImages.png', bbox_inches='tight')




# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)



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

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02
netG.apply(weights_init)

# Print the model
print(netG)




# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
  netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02
netD.apply(weights_init)

# Print the model
print(netD)




# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(100, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
lr_scheduler_D = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[50, 100, 150, 200, 250, 300, 350], gamma=0.5)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
lr_scheduler_G = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=[50, 100, 150, 200, 250, 300, 350], gamma=0.5)




# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
  progress_bar = qqdm(dataloader)
  # For each batch in the dataloader
  for i, data in enumerate(progress_bar):

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    netD.zero_grad()
    # Format batch
    real_cpu = data[0].to(device)
    b_size = real_cpu.size(0)
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
    # Forward pass real batch through D
    output = netD(real_cpu).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(b_size, nz, 1, 1, device=device)
    # Generate fake image batch with G
    fake = netG(noise)
    label.fill_(fake_label)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake).view(-1)
    # Calculate G's loss based on this output
    errG = criterion(output, label)
    # Calculate gradients for G
    errG.backward()
    # Update G
    optimizerG.step()

    # Output training stats\
    # Set the info of the progress bar
    # Note that the value of the GAN loss is not directly related to
    # the quality of the generated images.
    progress_bar.set_infos({
        'Loss_D': round(errD.item(), 4),
        'Loss_G': round(errG.item(), 4),
        'Epoch': epoch,
        'Step': iters,
    })


    # Save Losses for plotting later
    G_losses.append(errG.item())
    D_losses.append(errD.item())

    iters += 1

  # Save generated image with fixed noise in each epoch
  netG.eval()
  if(epoch % 20 == 0):
    f_imgs_sample = (netG(fixed_noise).data + 1) / 2.0
    filename = os.path.join(log_dir, f'Epoch_{epoch:03d}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
#   print(f' | Save some samples to {filename}.')

  # Show generated images in the notebook and save in img_list for later use.
  grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
  img_list.append(grid_img)
  plt.figure(figsize=(10,10))
  plt.imshow(grid_img.permute(1, 2, 0))
  plt.show()
  netG.train()

  # Save the checkpoints.
  if epoch % 5 == 0:
    torch.save(netG.state_dict(), os.path.join(ckpt_dir, 'netG.pth'))
    torch.save(netD.state_dict(), os.path.join(ckpt_dir, 'netD.pth'))

  lr_scheduler_D.step()
  lr_scheduler_G.step()


    




##########################################################################
# TODO: Plot the loss value of discriminator and generator
# Implementation A.1-2
##########################################################################
""" D & G's losses versus training iterations. """
plt.figure(figsize=(15, 15))
plt.xlabel("Iterations")
plt.ylabel('Loss')
plt.grid()

count = [i for i in range(len(G_losses))]
plt.plot( count, G_losses, color='red' , label='G',linewidth=2, markersize=10)
plt.plot( count, D_losses, color='blue', label='D',linewidth=2, markersize=10)


plt.legend(loc='lower right', prop={'size': 20})
plt.savefig('loss.png', bbox_inches='tight')


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig('RealFake.png', bbox_inches='tight')





# load gernerator
device = torch.device("cuda")
netG = Generator(ngpu)
netG.load_state_dict(torch.load(os.path.join(ckpt_dir, 'netG.pth')))
netG.eval()
netG.to(device)



# Generate 100 images and make a grid to save them.
n_output = 100
fixed_noise = torch.randn(100, nz, 1, 1, device=device)
imgs_sample = (netG(fixed_noise).data + 1) / 2.0
log_dir = os.path.join(workspace_dir, 'logs')
filename = os.path.join(log_dir, 'result.jpg')
torchvision.utils.save_image(imgs_sample, filename, nrow=10)

# Show 32 of the images in notebook.
grid_img = torchvision.utils.make_grid(imgs_sample[:32].cpu(), nrow=10)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.savefig('GenerateImage.png', bbox_inches='tight')




# Save the generated images in archive.
os.makedirs('output', exist_ok=True)
for i in range(n_output):
  torchvision.utils.save_image(imgs_sample[i], f'output/{i+1}.jpg')





##########################################################################
# TODO: Store your generate images in 5*5 Grid
# Implementation A.1-3
##########################################################################





##########################################################################
# TODO: Interpolation 3 pairs of z vectors and plot a 3*10 image.
# Implementation A.1-4
##########################################################################