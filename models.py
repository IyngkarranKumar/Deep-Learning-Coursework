import numpy as np
import torch
import pytorch_lightning as pl
import importlib
import config

from torch import nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid
import utils; importlib.reload(utils)



class Block(pl.LightningModule):
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

class ConvEncoder(nn.Module):

    def __init__(self,latent_size=512,in_f=3,img_dims=(32,32)):
        super().__init__()
        
        self.latent_size=latent_size
        self.in_f=in_f
        self.img_dims=img_dims

        #padding and sims
        self.log2_pad_dims=np.ceil(np.log2(img_dims[0])).astype('int')
        self.pad_dims=2**self.log2_pad_dims
        p1=((self.pad_dims-img_dims[0])/2).astype('int')
        self.pad_tens=(p1,p1,p1,p1) #pad last/second last axis above and below with p1 rows
        dims = [3]+list(2**np.arange(3,3+self.log2_pad_dims))
        self.dims=dims[:self.log2_pad_dims]+[self.latent_size]



        self.mu = nn.Linear(latent_size,latent_size)
        self.sigma = nn.Linear(latent_size,latent_size)
        self.encoder_layers=[]
        for dfrom,dto in zip(list(self.dims[:-1]),list(self.dims[1:])):
            self.encoder_layers.append(Block(dfrom,dto))
            self.encoder_layers.append(nn.MaxPool2d(kernel_size=(2,2)))
            

        
        self.encode=nn.Sequential(*self.encoder_layers)
        self.N = torch.distributions.Normal(0,1)
        

    def forward(self,x):

        #pad
        x=F.pad(input=x,pad=self.pad_tens,value=None)
        shape=x.shape
        x=self.encode(x).squeeze()
        mu=self.mu(x)
        sigma=torch.exp(self.sigma(x))
        z=mu+self.N.sample(mu.shape)*sigma
        return mu,sigma,z

class ConvDecoder(nn.Module):

    def __init__(self,latent_size=512,in_f=3,img_dims=(32,32)):
        super().__init__()
        
        self.latent_size=latent_size
        self.in_f=in_f
        self.img_dims=img_dims

        self.log2_pad_dims=np.ceil(np.log2(img_dims[0])).astype('int')
        self.pad_dims=2**self.log2_pad_dims
        p1=((self.pad_dims-img_dims[0])/2).astype('int')
        self.pad_tens=(p1,p1,p1,p1) #pad last/second last axis above and below with p1 rows
        dims = [3]+list(2**np.arange(3,3+self.log2_pad_dims))
        self.dims=list(reversed(dims[:self.log2_pad_dims]+[self.latent_size])) #reversed as decoding

        #yet to implement check for 'valid dimensions'!
        self.decoder_layers=[]
        for dto,dfrom in zip(list(self.dims[:-1]),list(self.dims[1:])):
            self.decoder_layers.append(nn.Upsample(scale_factor=2))
            self.decoder_layers.append(Block(in_f=dto,out_f=dfrom))

        self.decode=nn.Sequential(*self.decoder_layers)

    def forward(self,z):
        z=z[:,:,None,None]
        x_hat=self.decode(z)
        pd=tuple(-1*elem for elem in self.pad_tens) #to get rid of padding
        x_hat=F.pad(x_hat,pad=pd)
        return x_hat
        
class RegularVAELoss():

    def __call__(self,x,mu,sigma,z,x_hat,kl_weight=1):
        loss=F.mse_loss(x,x_hat) + (kl_weight)*utils.KLDivLoss(mu,sigma)
        return loss


class AbstractVariationalAutoencoder(pl.LightningModule):

    def __init__(self,encoder,decoder,loss_func,latent_size,**kwargs):
        super().__init__()

        self.encoder=encoder
        self.decoder=decoder
        self.loss_func=loss_func
        self.latent_size=latent_size

        self.warmup=kwargs.get('warmup')
        self.warmup_max_epoch=kwargs.get('warmup_max_epoch')
        print(self.warmup)
        self.warmup_schedule=np.linspace(0,1,self.warmup_max_epoch)
    
    def encode(self,x):
        mu,sigma,z=self.encoder(x)
        return mu,sigma,z

    def decode(self,z):
        x_hat=self.decoder(z)
        return x_hat

    def forward(self,x):
        mu,sigma,z=self.encode(x)
        x_hat=self.decode(z)
        return x_hat

    def training_step(self,batches,batch_idx):

        '''
        Accumulates loss over different training datasets (if multiple)
        '''

        loss=0
        for batch in batches.values():
            x,y=batch
            mu,sigma,z=self.encoder(x)
            x_hat=self.decoder(z)
            if self.warmup:
                schedule_idx=self.current_epoch if self.current_epoch<self.warmup_max_epoch else -1
                kl_weight=self.warmup_schedule[schedule_idx]
            else:
                kl_weight=1
            l=self.loss_func(x,mu,sigma,z,x_hat,kl_weight=kl_weight)
            loss+=l

        self.log('train_loss',loss,logger=True,on_epoch=True)

        return loss

    def validation_step(self,batches,batch_idx):
        loss=0
        for batch in batches.values():
            x,y=batch
            mu,sigma,z=self.encoder(x)
            x_hat=self.decoder(z)
            if self.warmup:
                schedule_idx=self.current_epoch if self.current_epoch<self.warmup_max_epoch else -1
                kl_weight=self.warmup_schedule[schedule_idx]
            else:
                kl_weight=1

            l=self.loss_func(x,mu,sigma,z,x_hat,kl_weight=kl_weight)
            loss+=l

        self.log('val_loss',loss,logger=True,on_epoch=True)

        return loss


    def configure_optimizers(self):
        optimiser=torch.optim.Adam(self.parameters(),lr=0.001)
        scheduler=torch.optim.lr_scheduler.StepLR(optimiser,step_size=100,gamma=0.5)
        return {'optimizer':optimiser,'lr_scheduler':scheduler}

    def sample(self,n_samples):

        prior=torch.distributions.Normal(0,1)
        z = prior.sample((n_samples,self.latent_size))

        with torch.no_grad():
            imgs=self.decoder(z)
            
        return imgs




class VAE(AbstractVariationalAutoencoder):
    
    def __init__(self,encoder=ConvEncoder(),decoder=ConvDecoder(),loss_func=RegularVAELoss(),latent_size=512,kwargs={}):
        super().__init__(encoder=encoder,decoder=decoder,loss_func=loss_func,latent_size=latent_size,**kwargs)



    




#working VAE
class VariationalAutoencoder(pl.LightningModule):

    def __init__(self,latent_size=512,in_f=3,img_dims=(32,32),device=torch.device('cpu')):
        super(VariationalAutoencoder,self).__init__()


        self.latent_size=latent_size
        self.in_f=in_f
        self.img_dims=img_dims
        self.dev=device
        self.encoder_layers=int(np.log2(img_dims[0])) #sets encoder ASSUMING each layer halves img dims (2n,2n)->(n,n)
        self.dims=list(reversed([latent_size/(2**i) for i in range(self.encoder_layers)]))
        self.dims = np.array([in_f]+self.dims,dtype=int)
        self.mu = nn.Linear(latent_size,latent_size)
        self.sigma = nn.Linear(latent_size,latent_size)
        self.encoder_layers=[]
        self.decoder_layers=[]
        for dfrom,dto in zip(list(self.dims[:-1]),list(self.dims[1:])):
            self.encoder_layers.append(Block(dfrom,dto))
            self.encoder_layers.append(nn.MaxPool2d(kernel_size=(2,2)))
        for dto,dfrom in zip(list(reversed(self.dims[:-1])),list(reversed(self.dims[1:]))):
            self.decoder_layers.append(nn.Upsample(scale_factor=2))
            self.decoder_layers.append(Block(dfrom,dto))
        self.encode=nn.Sequential(*self.encoder_layers)
        self.decode=nn.Sequential(*self.decoder_layers)
        self.N = torch.distributions.Normal(0,1)

        self.save_hyperparameters() #for checkpointing

    def encoder(self,x):
        x=self.encode(x).squeeze()
        mu = self.mu(x)
        sigma=torch.exp(self.sigma(x))
        z = mu + sigma*(self.N.sample(mu.shape).type_as(mu))

        return mu,sigma,z

    def decoder(self,z):
        z = z[:,:,None,None]
        x_hat=self.decode(z)
        
        return x_hat


    def forward(self,x):
        mu,sigma,z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat

    def training_step(self,batch,batch_idx):
        x,y=batch
        mu,sigma,z=self.encoder(x)
        x_hat=self.decoder(z)
        loss = F.mse_loss(x,x_hat) + utils.KLDivLoss(mu,sigma);
        self.log('train_loss',loss,logger=True,on_epoch=True)

        return loss

    def validation_step(self,batch,batch_idx):
        x,y=batch
        mu,sigma,z=self.encoder(x)
        x_hat=self.decoder(z)
        loss=F.mse_loss(x,x_hat) + utils.KLDivLoss(mu,sigma)
        self.log('val_loss',loss,logger=True,on_epoch=True)

        return loss
        

    def configure_optimizers(self):
        optimiser=torch.optim.Adam(self.parameters(),lr=0.001)
        scheduler=torch.optim.lr_scheduler.StepLR(optimiser,step_size=100,gamma=0.5)
        return {'optimizer':optimiser,'lr_scheduler':scheduler}

    def on_train_batch_end(self, outputs,batch,batch_idx):
        return batch

    def sample(self,n_samples=5,method='random'):
        #sample from prior

        prior=torch.distributions.Normal(0,1)
        z = prior.sample((n_samples,self.latent_size)).to(self.dev)
        prior.sample()

        with torch.no_grad():
            imgs=self.decoder(z)
            
        return imgs