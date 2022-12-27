
import torch
import os


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

def FID():
  #for calculation of FID score
  pass