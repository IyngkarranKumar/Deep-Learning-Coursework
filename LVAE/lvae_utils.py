
import torch

class LVAE():
    
    def __init__(self,n_layers):
        
        self.N = torch.distributions.Normal(0,1)


    def bottom_up_pass(X):
        pass

    def top_down_pass(X):
        pass



    def forward(self,x):

        inf_mus,inf_sigmas,zs=bottom_up_pass(X)
        inf_mus,inf_sigmas,gen_mus,gen_sigmas,x_hat=top_down_pass(X)
        return inf_mus,inf_sigmas,gen_mus,gen_sigmas,z,x_hat

    def training_step(self,batch,batch_idx):

        X,y=batch
        inf_mus,inf_sigmas,gen_mus,gen_sigmas,z,x_hat=self.forward(X)
        loss=MSE(X,x_hat)+KL_divergence(inf_mus,inf_sigmas,)


