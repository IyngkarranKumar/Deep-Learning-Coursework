import importlib
import utils
import torch
import pytorch_lightning as pl
import os
import torchvision

from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
importlib.reload(utils)

#datasets here split Train dataset 80:20 into train and val

class CIFAR10Dataset(pl.LightningDataModule):

    def __init__(self,train_path='training',test_path='test',batch_size=64,img_size=None):

        self.train_path=train_path
        self.test_path=test_path
        self.batch_size=batch_size
        self.image_size=img_size

    def prepare_data(self):
        #download data
        torchvision.datasets.CIFAR10(self.train_path,train=True,download=True)
        torchvision.datasets.CIFAR10(self.test_path,train=False,download=True)


    def setup(self):
        #apply transforms and splits
        transform_list=[torchvision.transforms.ToTensor()]
        if self.image_size is not None:
            transform_list.append(transforms.Resize(self.image_size))

        transforms = torchvision.transforms.Compose(transform_list)
        

        cifar10_full = torchvision.datasets.CIFAR10(self.train_path,train=True,transform=transforms)
        self.cifar10_train,self.cifar10_val = random_split(cifar10_full,[int(0.8*cifar10_full.__len__()),int(0.2*cifar10_full.__len__())])

        self.cifar10_test = torchvision.datasets.CIFAR10(self.test_path,train=False,transform=transforms)

    def train_dataloader(self,num=None):
        return DataLoader(self.cifar10_train,batch_size=self.batch_size)


    def test_dataloader(self):
        return DataLoader(self.cifar10_test,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val,batch_size=self.batch_size)


class STL10Dataset(pl.LightningDataModule):
    
    def __init__(self,batch_size=64,img_size=None):
        self.train_path='training/stl10' #path to download
        self.test_path='test/stl10'
        self.batch_size=batch_size
        self.image_size=(32,32)

    def prepare(self):
        torchvision.datasets.STL10(self.train_path,train=True,download=True)
        torchvision.datasets.STL10(self.test_path,train=False,download=True)

    def setup(self):

        #apply transforms and splits
        transform_list=[torchvision.transforms.ToTensor()]
        if self.image_size is not None:
            transform_list.append(transforms.Resize(self.image_size))

        transforms = torchvision.transforms.Compose(transform_list)
        

        stl10_full = torchvision.datasets.STL10(self.train_path,train=True,transform=transforms)
        self.stl10_train,self.stl10_val = random_split(stl10_full,[int(0.8*stl10_full.__len__()),int(0.2*stl10_full.__len__())])

        self.stl10_test = torchvision.datasets.STL10(self.test_path,train=False,transform=transforms)

    def train_dataloader(self):
        return DataLoader(self.stl10_train,batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.stl10_test,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.stl10_val,batch_size=self.batch_size)


class LSUNDataset(pl.LightningDataModule):
    
    def __init__(self,batch_size=64,img_size=None):
        self.train_path='training/lsun' #path to download
        self.test_path='test/lsun'
        self.batch_size=batch_size
        self.image_size=(32,32)

    def prepare(self):
        torchvision.datasets.LSUN(self.train_path,train=True,download=True)
        torchvision.datasets.LSUN(self.test_path,train=False,download=True)

    def setup(self):

        #apply transforms and splits
        transform_list=[torchvision.transforms.ToTensor()]
        if self.image_size is not None:
            transform_list.append(transforms.Resize(self.image_size))

        transforms = torchvision.transforms.Compose(transform_list)
        

        lsun_full = torchvision.datasets.LSUN(self.train_path,train=True,transform=transforms)
        self.lsun_train,self.lsun_val = random_split(lsun_full,[int(0.8*lsun_full.__len__()),int(0.2*lsun_full.__len__())])

        self.lsun_test = torchvision.datasets.lsun(self.test_path,train=False,transform=transforms)

    def train_dataloader(self):
        return DataLoader(self.lsun_train,batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.lsun_test,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.lsun_val,batch_size=self.batch_size)
