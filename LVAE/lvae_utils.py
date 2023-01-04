
import torch

class LVAE():
    
    def __init__(self):
        
        self.N = torch.distributions.Normal(0,1)

    def forward(self,x):
        
        #bottom up pass - find parameters for inference model
        for layer in layers:
            mu_i,sigma_i=mu_i(x),sigma_i(x)
            z=mu_i+sigma_i*self.N.sample()

        z=z_top #top level latent

        #top down pass - find parameters for generator model and update parameters for inference models
