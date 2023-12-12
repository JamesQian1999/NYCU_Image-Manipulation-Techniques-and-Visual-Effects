# Import packages
import random
import numpy as np
import cv2
import einops
import imageio
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale
from torchvision.datasets import ImageFolder
import torchvision
import os
# from IPython.display import Image


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


dataset       = "anime" 
workspace_dir = 'Final'
os.makedirs(f'{dataset}_{workspace_dir}',exist_ok=False)

dataset_path = f"anime_face_dataset"
# The path to save the model
model_store_path = f"{dataset}_{workspace_dir}/anime.pt"
milestones = [1000, 2000]
image_shape = (3,64,64)
save = 50

os.makedirs(f'{dataset}_{workspace_dir}/log',exist_ok=False)

# Batch size during training
batch_size = 512

# Number of training epochs
n_epochs = 401

# Learning rate for optimizers
lr = 8e-5

# Number of the forward steps
n_steps = 1000

# Initial beta
start_beta = 1e-4

# End beta
end_beta = 1e-2


f = open(f"{dataset}_{workspace_dir}/hyperparameter.txt", 'w')
f.write(f'model_store_path: {str(model_store_path)}\n')
f.write(f'batch_size: {str(batch_size)}\n')
f.write(f'n_epochs: {str(n_epochs)}\n')
f.write(f'lr: {str(lr)}\n')
f.write(f'n_steps: {str(n_steps)}\n')
f.write(f'start_beta: {str(start_beta)}\n')
f.write(f'end_beta: {str(end_beta)}\n')
f.close()

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# List to keep track of loss
loss_list = []

trans = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)]
)
datasets = ImageFolder(root=dataset_path, transform=trans)

# Make the data loader
dataloader = DataLoader(datasets, batch_size, shuffle=True, num_workers=12)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig(f'{dataset}_{workspace_dir}/TrainingImages.png', bbox_inches='tight')


# Show images
def show_images(images, title=""):
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, fontsize=24)
    rows = int(len(images) ** 0.5)
    cols = round(len(images) / rows)
    index = 0
    for row in range(rows):
        for col in range(cols):
            fig.add_subplot(rows, cols, index + 1)
            if index < len(images):
                frame = plt.gca()
                frame.axes.get_yaxis().set_visible(False)
                frame.axes.get_xaxis().set_visible(False)
                temp = np.transpose(images[index], (1, 2, 0))
                plt.imshow((temp+1)/2, cmap='gray' if images[index].shape[0]==1 else None)
                index += 1
    plt.show()

# Show images of next batch
def show_images_of_next_batch(loader):
    dataiter = iter(dataloader)
    data = next(dataiter)
    features, labels = data
    show_images(features, "Images in a batch")

show_images_of_next_batch(dataloader)


# Define the class of DDPM
class DDPM(nn.Module):
    def __init__(self, image_shape=(1, 28, 28), n_steps=200, start_beta=1e-4, end_beta=0.02, device=None):
        super(DDPM, self).__init__()
        self.device = device
        self.image_shape = image_shape
        self.n_steps = n_steps
        self.noise_predictor = UNet(n_steps, shape = image_shape[-1], channel = image_shape[0]).to(device)
        self.betas = torch.linspace(start_beta, end_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    # Forward process
    # Add the noise to the images
    def forward(self, x0, t, eta=None):
        n, channel, height, width = x0.shape
        alpha_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, channel, height, width).to(self.device)

        noise = alpha_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - alpha_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noise

    # Backward process
    # Predict the noise that was added to the images during the forward process
    def backward(self, x, t):
        return self.noise_predictor(x, t)


# Create the time embedding
def time_embedding(n, d):
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])
    return embedding


# Define the class of U-Net  
class UNet(nn.Module):
    def __init__(self, n_steps=1000, time_embedding_dim=256, shape = 28, channel = 1):
        super(UNet, self).__init__()

        # Time embedding
        self.time_step_embedding = nn.Embedding(n_steps, time_embedding_dim)
        self.time_step_embedding.weight.data = time_embedding(n_steps, time_embedding_dim)
        self.time_step_embedding.requires_grad_(False)

        # The first half
        self.time_step_encoder1 = nn.Sequential(
            nn.Linear(time_embedding_dim, 1),
            nn.SiLU(),
            nn.Linear(1, 1)
        )

        self.block1 = nn.Sequential(
            nn.LayerNorm((channel, shape, shape)),
            nn.Conv2d(channel, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.down1 = nn.Conv2d(8, 8, 4, 2, 1)

        self.time_step_encoder2 = nn.Sequential(
            nn.Linear(time_embedding_dim, 8),
            nn.SiLU(),
            nn.Linear(8, 8)
        )

        self.block2 = nn.Sequential(
            nn.LayerNorm((8, shape//2, shape//2)),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.down2 = nn.Conv2d(16, 16, 4, 2, 1)

        self.time_step_encoder3 = nn.Sequential(
            nn.Linear(time_embedding_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 16)
        )

        self.block3 = nn.Sequential(
            nn.LayerNorm((16, shape//4, shape//4)),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, 2, 1),
        )

        # The bottleneck
        self.time_step_encoder_mid = nn.Sequential(
            nn.Linear(time_embedding_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )

        self.block_mid = nn.Sequential(
            nn.LayerNorm((32, shape//8, shape//8)),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        # The second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 3, 1, 1)
        )

        self.time_step_encoder4 = nn.Sequential(
            nn.Linear(time_embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

        self.block4 = nn.Sequential(
            nn.LayerNorm((64, shape//4, shape//4)),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.up2 = nn.ConvTranspose2d(16, 16, 4, 2, 1)

        self.time_step_encoder5 = nn.Sequential(
            nn.Linear(time_embedding_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )

        self.block5 = nn.Sequential(
            nn.LayerNorm((32, shape//2, shape//2)),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.up3 = nn.ConvTranspose2d(8, 8, 4, 2, 1)

        self.time_step_encoder6 = nn.Sequential(
            nn.Linear(time_embedding_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 16)
        )
        self.block6 = nn.Sequential(
            nn.LayerNorm((16, shape, shape)),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.LayerNorm((8, shape, shape)),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.LayerNorm((8, shape, shape)),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            
        )

        self.final_layer = nn.Conv2d(8, channel, 3, 1, 1)

    def forward(self, x, t):
        t = self.time_step_embedding(t)
        n = len(x)
        output1 = self.block1(x + self.time_step_encoder1(t).reshape(n, -1, 1, 1))
        output2 = self.block2(self.down1(output1) + self.time_step_encoder2(t).reshape(n, -1, 1, 1))
        output3 = self.block3(self.down2(output2) + self.time_step_encoder3(t).reshape(n, -1, 1, 1))

        output_mid = self.block_mid( self.down3(output3) + self.time_step_encoder_mid(t).reshape(n, -1, 1, 1))

        output4 = torch.cat((output3, self.up1(output_mid)), dim=1)
        output4 = self.block4(output4 + self.time_step_encoder4(t).reshape(n, -1, 1, 1))
        output5 = torch.cat((output2, self.up2(output4)), dim=1)
        output5 = self.block5(output5 + self.time_step_encoder5(t).reshape(n, -1, 1, 1))
        output6 = torch.cat((output1, self.up3(output5)), dim=1)
        output6 = self.block6(output6 + self.time_step_encoder6(t).reshape(n, -1, 1, 1))

        output = self.final_layer(output6)
        return output
   

# Build the DDPM
ddpm_mnist = DDPM(image_shape=image_shape, n_steps=n_steps, start_beta=start_beta, end_beta=end_beta, device=device)

# Print the model
print(ddpm_mnist)


# Sample the first image from the next batch, then demonstrate the forward process.
def show_forward(ddpm, loader, device):
    fig = plt.figure(figsize=(6, 1))

    for batch in loader:

        images = batch[0]
        fig.add_subplot(161)
        temp = np.transpose(images[0], (1, 2, 0))
        plt.title('original')
        plt.imshow((temp+1)/2, cmap='gray')
        plt.axis('off')

        tensor_image = ddpm(images[:1].to(device), [int(0.1 * ddpm.n_steps) - 1])
        image = tensor_image.detach().cpu().numpy()
        fig.add_subplot(162)
        temp = np.transpose(image[0], (1, 2, 0))
        plt.title('10%')
        plt.imshow((temp+1)/2, cmap='gray')
        plt.axis('off')

        tensor_image = ddpm(images[:1].to(device), [int(0.25 * ddpm.n_steps) - 1])
        image = tensor_image.detach().cpu().numpy()
        fig.add_subplot(163)
        temp = np.transpose(image[0], (1, 2, 0))
        plt.title('25%')
        plt.imshow((temp+1)/2, cmap='gray')
        plt.axis('off')

        tensor_image = ddpm(images[:1].to(device), [int(0.5 * ddpm.n_steps) - 1])
        image = tensor_image.detach().cpu().numpy()
        fig.add_subplot(164)
        temp = np.transpose(image[0], (1, 2, 0))
        plt.title('50%')
        plt.imshow((temp+1)/2, cmap='gray')
        plt.axis('off')

        tensor_image = ddpm(images[:1].to(device), [int(0.75 * ddpm.n_steps) - 1])
        image = tensor_image.detach().cpu().numpy()
        fig.add_subplot(165)
        temp = np.transpose(image[0], (1, 2, 0))
        plt.title('75%')
        plt.imshow((temp+1)/2, cmap='gray')
        plt.axis('off')

        tensor_image = ddpm(images[:1].to(device), [int(1 * ddpm.n_steps) - 1])
        image = tensor_image.detach().cpu().numpy()
        fig.add_subplot(166)
        temp = np.transpose(image[0], (1, 2, 0))
        plt.title('100%')
        plt.imshow((temp+1)/2, cmap='gray')
        plt.axis('off')
        break


"""
Provided with a DDPM model, a specified number of samples to generate, and a chosen device,
this function returns a set of freshly generated samples while also saving the .gif of the reverse process
"""
def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=25, gif_name="sampling.gif", channel=1, height=28, width=28):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, channel, height, width).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, channel, height, width).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

    if channel == 1:
        for i in range(len(frames)):
            frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(10,10))
    plt.imshow(frames[-1])
    plt.savefig(f'{gif_name.split(".")[0]}.png', bbox_inches='tight')

    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frames[-1])
    return x

def trainer(ddpm, dataloader, n_epochs, optim, loss_funciton, device, model_store_path):

    f = open(f"{dataset}_{workspace_dir}/loss.txt", 'a')
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="green"):
        epoch_loss = 0.0
        progress = tqdm( enumerate(dataloader), leave=False, colour="blue", total=len(dataloader))
        for step, batch in progress:
            # Load data
            x0 = batch[0].to(device)
            n = len(x0)
            # Pick random noise for each of the images in the batch
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Compute the noisy image based on x0 and the time step
            noises = ddpm(x0, t, eta)

            # Get model estimation of noise based on the images and the time step
            eta_theta = ddpm.backward(noises, t.reshape(n, -1))

            # Optimize the Mean Squared Error (MSE) between the injected noise and the predicted noise
            loss = loss_funciton(eta_theta, eta)

            # First, initialize the optimizer's gradient and then update the network's weights
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Aggregate the loss values from each iteration to compute the loss value for an epoch
            epoch_loss += loss.item() * len(x0) / len(dataloader.dataset)

            # Save Losses for plotting later
            loss_list.append(loss.item())
            update_txt = f" | Loss: {loss.item():.4f}"
            f.write(f'{str(loss.item())}\n')

            progress.set_postfix_str(update_txt, refresh=True)

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        if epoch % save == 0:
            generate_new_images(ddpm, device=device, gif_name=f"{dataset}_{workspace_dir}/log/epoch{epoch}.gif", height=image_shape[-1], width=image_shape[-1], channel=image_shape[0])

        # If the current loss is better than the previous one, then store the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), model_store_path)
            log_string += " <Store the best model.>"

    f.close()

optimizer = optim.Adam(ddpm_mnist.parameters(), lr)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5) 
trainer(ddpm_mnist, dataloader, n_epochs=n_epochs, optim=optimizer, loss_funciton=nn.MSELoss(), device=device, model_store_path=model_store_path) 

# Build the model
ddpm_mnist = DDPM(image_shape=image_shape, n_steps=n_steps, device=device)

# Load the state of the trained model
ddpm_mnist.load_state_dict(torch.load(model_store_path, map_location=device))

# Change to evaluation mode
ddpm_mnist.eval()

images = generate_new_images(
        ddpm_mnist, 
        n_samples = 100,
        device = device, 
        gif_name=f"{dataset}_{workspace_dir}/{dataset}.gif",
        height=image_shape[-1], width=image_shape[-1], channel=image_shape[0])


######################################################################################
# TODO: Plot the loss values of DDPM 
######################################################################################
plt.figure(figsize=(15, 15))
plt.xlabel("Iterations")
plt.ylabel('Loss')
plt.grid()

count = [i for i in range(len(loss_list))]
plt.plot( count, loss_list, color='blue', label='D',linewidth=2, markersize=10)

plt.legend(loc='lower right', prop={'size': 20})
plt.savefig(f'{dataset}_{workspace_dir}/loss_anime.png', bbox_inches='tight')

######################################################################################
# TODO: Store your generate images in 5*5 grid
######################################################################################
images = generate_new_images(
        ddpm_mnist, 
        n_samples = 25,
        device = device, 
        gif_name=f"{dataset}_{workspace_dir}/result_anime.gif",
        height=image_shape[-1], width=image_shape[-1], channel=image_shape[0])


