import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets, transforms
import torch.nn.utils.spectral_norm as spectral_norm
import os

# Self-Attention Module Placeholder
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key =  self.key(x).view(m_batchsize,-1,width*height)
        energy =  torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        x += self.gamma*out
        return x

class ResidualGenerator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_g):
        super(ResidualGenerator, self).__init__()
        self.init_size = 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, feature_g * 8 * self.init_size ** 2))

        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(feature_g * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(feature_g * 8, feature_g * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(feature_g * 4, 0.8),
            nn.ReLU(inplace=True),
            SelfAttention(feature_g * 4),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(feature_g * 4, feature_g * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(feature_g * 2, 0.8),
            nn.ReLU(inplace=True),
            SelfAttention(feature_g * 2),

            nn.Conv2d(feature_g * 2, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class ResidualDiscriminator(nn.Module):
    def __init__(self, img_channels, feature_d):
        super(ResidualDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1))]
            if bn:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(img_channels, feature_d, bn=False), 
            *discriminator_block(feature_d, feature_d * 2),
            *discriminator_block(feature_d * 2, feature_d * 4),
            *discriminator_block(feature_d * 4, feature_d * 8),
            spectral_norm(nn.Conv2d(feature_d * 8, 1, 1, stride=1, padding=0))  
        )
        
    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)

class MNISTUbyteDataset(Dataset):
    def _init_(self, images_path, labels_path, transform=None):
        self.images = self.load_images(images_path)
        self.labels = self.load_labels(labels_path) 
        self.transform = transform

    def load_images(self, path):
        with open(path, 'rb') as f:
            f.read(16)
            data = np.frombuffer(f.read(), np.uint8).astype(np.float32)
            data = data.reshape(-1, 28, 28)
            return data / 255.0

    def load_labels(self, path):
        with open(path, 'rb') as f:
            f.read(8)
            return np.frombuffer(f.read(), np.uint8)

    def _len_(self):
        return len(self.labels)

    def _getitem_(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = np.expand_dims(image, axis=0)

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32)

        return image, label
    
class FashionMNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data = datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label
    
class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

# Evaluate Function Placeholder
def evaluate(generator, latent_dim, device, output_dir="generated_images", n_images=16):
    generator.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        z = torch.randn(n_images, latent_dim, device=device)
        
        gen_imgs = generator(z)

        save_path = os.path.join(output_dir, "generated_grid.png")
        vutils.save_image(gen_imgs, save_path, normalize=True, nrow=4)

    print(f"Generated images saved to {save_path}")
    generator.train()

# GAN Training Loop
# Remaining code for training, including initializing the models, loss function, optimizer, and dataloader...
