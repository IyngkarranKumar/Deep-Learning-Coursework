
import torch
import os
import matplotlib.pyplot as plt
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def normalise(x):
  return (x-x.min()) / (x.max()-x.min())

def save(save_dict,name):
  path = os.path.join(os.getcwd(),"models",name)
  torch.save(save_dict,path)

def load(path,model,optimiser):

  checkpoint = torch.load(path)
  model_state = model.load_state_dict(checkpoint["model_state_dict"])
  optimiser_state = optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
  epoch = checkpoint["epoch"]
  loss = checkpoint["loss"]

  return model,optimiser,epoch,loss


def KLDivLoss(mu1,sigma1,mu2=0,sigma2=1):
  #code from https://github.com/robert-lieck/Learning-Embeddings-by-Simulating-Communication/blob/main/CommunicationAutoencoder/util.py
  
  loss = (0.5*torch.log(sigma2/sigma1) + ((sigma1**2 + (mu1-mu2)**2) / (2*sigma2**2)) - 0.5).mean()
  return loss

def FID(D_real,D_gen):
  '''
  Idea behind FID - two datasets, one real one fake. D_R, D_F.
  Calculate activation statistics for D_R and D_F both passed into InceptionV3 Network.
  Calculate difference in resulting distributions
  '''

  fid=FrechetInceptionDistance(normalize=True)
  fid.update(D_real,real=True)
  fid.update(D_gen,real=False)
  fid_score=fid.compute()
  return fid_score

def num_nans_infs(tens):
  num_nans=len(torch.nonzero(torch.isnan(tens.view(-1))))
  num_infs=len(torch.nonzero(torch.isinf(tens.view(-1))))
  print('Number nans/infs: {},{}'.format(num_nans,num_infs))
  
  return

def imshow(tens):
  tens=tens.permute(1,2,0)
  plt.imshow(tens.detach().numpy())
  ax=plt.gca()
  return ax




