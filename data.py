import importlib
import utils
import torch
import pytorch_lightning as pl
import os
import torchvision

from torch.utils.data import Dataset,DataLoader,random_split
importlib.reload(utils)


class CIFAR10Dataset(pl.LightningDataModule):

    def __init__(self,batch_size=64):

        self.train_path='training/cifar10'
        self.test_path='test/cifar10'
        self.batch_size=batch_size

    def prepare_data(self):
        #download data
        torchvision.datasets.CIFAR10(self.train_path,train=True,download=True)
        torchvision.datasets.CIFAR10(self.test_path,train=False,download=True)


    def setup(self):
        
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        cifar10_full = torchvision.datasets.CIFAR10(self.train_path,train=True,transform=transforms)
        self.cifar10_train,self.cifar10_val = random_split(cifar10_full,[int(0.8*cifar10_full.__len__()),int(0.2*cifar10_full.__len__())])

        self.cifar10_test = torchvision.datasets.CIFAR10(self.test_path,train=False,transform=transforms)

    def train_dataloader(self,num=None):
        return DataLoader(self.cifar10_train,batch_size=self.batch_size)


    def test_dataloader(self):
        return DataLoader(self.cifar10_test,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val,batch_size=self.batch_size)
