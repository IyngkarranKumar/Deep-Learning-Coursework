#%%
import numpy as np
import torch
import pytorch_lightning as pl
import importlib
import matplotlib.pyplot as plt
%matplotlib inline
import torchvision
import os
import pickle

from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
import data; importlib.reload(data)
import models; importlib.reload(models)
import utils; importlib.reload(utils)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%% setup data
import data; importlib.reload(data)

vis=False

dataset = data.CIFAR10Dataset()

#download data
dataset.prepare_data()

#transform etc.
dataset.setup()

#dataloaders
train_dataloader = dataset.train_dataloader(num=300)
train_iterator = iter(utils.cycle(train_dataloader))
X_toy,Y_toy = train_iterator.__next__()

h,w,n_channels = dataset.cifar10_train.dataset.data.shape[1:]


if vis:
    x,y=next(train_iterator)
    x,y=x.to(device),y.to(device)
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(x[:32]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)

#%% Loading models
import models; importlib.reload(models)

load=False; load_path='models/Model 1/lightning_logs/version_0/checkpoints/epoch=1-step=10.ckpt'

if 0:
    model_dict = {
        'model type':type(models.Autoencoder),
        'args':args_dict

    }

    if model_dict['model type'] == type(models.Autoencoder):
        kwargs=model_dict['args']
        model = models.Autoencoder(**kwargs)

    
latent_size=512

if not load:
    model = models.Autoencoder(n_channels=n_channels,latent_size=latent_size)



#%% Training model
import models;importlib.reload(models)

model_num = len(os.listdir('models'))+1
name=' '.join(['Model',str(model_num)])
checkpoint_path = os.path.join('models',name)

if 1: #callbacks
    
    class SamplesCallback(Callback):

        def __init__(self,sample_freq=2,n_samples=5):
            self.sample_freq=sample_freq
            self.n_samples=n_samples
            self.final_sample=None

        #return batch
        #def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
            #return batch

        #show samples 
        def on_train_epoch_end(self,trainer,model):
            if (trainer.current_epoch% self.sample_freq==0) and (trainer.current_epoch!=0):
                imgs=utils.normalise(model.sample(n_samples=self.n_samples))
                #real_imgs= self.on_train_batch_end()[:self.n_samples]
                #imgs=torch.concat([imgs,real_imgs])
                grid = make_grid(imgs.cpu().data,nrow=self.n_samples).permute(1,2,0)
                
                fig,ax = plt.subplots(figsize=(6,6))
                ax.imshow(grid)
                plt.show()

                self.sample=grid


    #simple test callback
    class PrintCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            print("Training is started!")
        def on_train_epoch_end(self,trainer,model):
            print('\n {}'.format(trainer.current_epoch))
        def on_train_end(self, trainer, pl_module):
            print("Training is done.")

save=True
max_epochs=3
callbacks = [SamplesCallback()] #custom code added to pl Trainer

if save:
    trainer=pl.Trainer(max_epochs=max_epochs,callbacks=callbacks,default_root_dir=checkpoint_path)
else:
    trainer=pl.Trainer(max_epochs=max_epochs,callbacks=callbacks)

trainer.fit(model,train_dataloader)

#saving images and extra model details
if save:
    final_sample=trainer.callbacks[0].sample

    #pickle file
    details={
        'model type':type(str(model)),
        'final sample':final_sample
    }
    details_path=os.path.join(checkpoint_path,'details.pkl')
    with open(details_path,'wb') as f:
        pickle.dump(details,f)

    #image
    image_path=os.path.join(checkpoint_path,'final_sample.png')
    fig=plt.figure();plt.imshow(final_sample)
    plt.savefig(image_path)
    plt.close(fig)


#%% test

import models;importlib.reload(models)

VAE = models.VariationalAutoencoder()
out = VAE(X_toy)