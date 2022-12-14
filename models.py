import numpy as np
import torch
import pytorch_lightning as pl
import importlib


from torch import nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid
import utils; importlib.reload(utils)


class Block(nn.Module):
    #maps (in_f,2n,2n) -> (out_f,2n,2n)
    def __init__(self, in_f, out_f):
        super(Block, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        return self.f(x)

class Autoencoder(pl.LightningModule):
    
    def __init__(self,n_channels,latent_size, f=16):
        super().__init__()

        self.latent_size=latent_size
        self.n_channels=n_channels
        self.f=16

        self.save_hyperparameters() #for checkpointing

        self.encode = nn.Sequential(
            Block(n_channels, f),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 16x16 (if cifar10, 48x48 if stl10)
            Block(f  ,f*2),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 8x8
            Block(f*2,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 4x4
            Block(f*4,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 2x2
            Block(f*4,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 1x1
            Block(f*4,latent_size),
        )

        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2), # output = 2x2
            Block(latent_size,f*4),
            nn.Upsample(scale_factor=2), # output = 4x4
            Block(f*4,f*4),
            nn.Upsample(scale_factor=2), # output = 8x8
            Block(f*4,f*2),
            nn.Upsample(scale_factor=2), # output = 16x16
            Block(f*2,f  ),
            nn.Upsample(scale_factor=2), # output = 32x32
            nn.Conv2d(f,n_channels, 3,1,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        shape = x.shape
        z = self.encode(x)
        x_hat = self.decode(z)
        return z,x_hat

    def training_step(self,batch,batch_idx):
        x,y=batch
        z,x_hat=self(x)
        loss = torch.nn.functional.mse_loss(x,x_hat)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def on_train_batch_end(self, outputs,batch,batch_idx):
        return batch

    def sample(self,n_samples=5,method='random'):

        if method=='random':
            z=torch.rand(n_samples,self.latent_size)
            z=z[:,:,None,None]

        with torch.no_grad():
            imgs=self.decode(z)
            
        return imgs

class VariationalAutoencoder(pl.LightningModule):

    def __init__(self,latent_size=512,in_f=3,img_dims=(32,32)):
        super(VariationalAutoencoder,self).__init__()


        self.latent_size=latent_size
        self.in_f=in_f
        self.img_dims=img_dims
        self.encoder_layers=int(np.log2(img_dims[0])) #sets encoder ASSUMING each layer halves img dims (2n,2n)->(n,n)
        self.dims=list(reversed([latent_size/(2**i) for i in range(self.encoder_layers)]))
        self.dims = np.array([in_f]+self.dims,dtype=int)
        self.encoder_layers=[]
        self.decoder_layers=[]
        self.mu = nn.Linear(latent_size,latent_size)
        self.sigma = nn.Linear(latent_size,latent_size)


        self.encoder=nn.ModuleList(self.encoder_layers)
        self.decoder=nn.ModuleList(self.decoder_layers)
        self.N = torch.distributions.Normal(0,1)

    def encoder(self,x):
        
        for dfrom,dto in zip(self.dims[:-1],self.dims[1:]):
            x = Block(dfrom,dto)(x)
            x = nn.MaxPool2d(kernel_size=(2,2))(x)
        x=x.squeeze()
        mu = self.mu(x)
        sigma=torch.exp(self.sigma(x))
        z = mu + sigma*self.N.sample(mu.shape)

        return mu,sigma,z

    def decoder(self,z):
        z = z[:,:,None,None]
        for dto,dfrom in zip(list(reversed(self.dims[:-1])),list(reversed(self.dims[1:]))):
            z = nn.Upsample(scale_factor=2)(z)
            z = Block(dfrom,dto)(z)
        x_hat=z
        
        return x_hat


    def forward(self,x):
        mu,sigma,z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat

    def training_step(self,batch,batch_idx):
        x,y=batch
        mu,sigma,z=self.encoder(x)
        x_hat=self.decoder(z)
        loss = F.mse_loss(x,x_hat) + utils.KLDivLoss(mu,sigma)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def on_train_batch_end(self, outputs,batch,batch_idx):
        return batch

    def sample(self,n_samples=5,method='random'):
        #sample from prior

        prior=torch.distributions.Normal(0,1)
        z = prior.sample((n_samples,self.latent_size))
        prior.sample()

        with torch.no_grad():
            imgs=self.decoder(z)
            
        return imgs