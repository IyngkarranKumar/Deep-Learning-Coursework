import importlib
import utils
import torch
import pytorch_lightning as pl
import os
import torchvision

from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
from torch.utils.data import ConcatDataset
importlib.reload(utils)

#datasets here split Train dataset 80:20 into train and val

class CIFAR10Dataset(pl.LightningDataModule):

    def __init__(self,train_path='data/cifar10/train',test_path='data/cifar10/test',batch_size=64,img_size=None):

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

        img_transform = torchvision.transforms.Compose(transform_list)
        

        cifar10_full = torchvision.datasets.CIFAR10(self.train_path,train=True,transform=img_transform)
        self.cifar10_train,self.cifar10_val = random_split(cifar10_full,[int(0.8*cifar10_full.__len__()),int(0.2*cifar10_full.__len__())])

        self.cifar10_test = torchvision.datasets.CIFAR10(self.test_path,train=False,transform=img_transform)

    def train_dataloader(self,num=None):
        return DataLoader(self.cifar10_train,batch_size=self.batch_size)


    def test_dataloader(self):
        return DataLoader(self.cifar10_test,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val,batch_size=self.batch_size)


class STL10Dataset(pl.LightningDataModule):
    
    def __init__(self,train_path='data/stl10/train',test_path='data/stl10/test',batch_size=64,img_size=None):
        self.train_path=train_path #path to download
        self.test_path=test_path
        self.batch_size=batch_size
        self.image_size=img_size

    def prepare_data(self):
        torchvision.datasets.STL10(self.train_path,split='train',download=True)
        torchvision.datasets.STL10(self.test_path,split='test',download=True)

    def setup(self):

        #apply transforms and splits
        transform_list=[torchvision.transforms.ToTensor()]
        if self.image_size is not None:
            transform_list.append(transforms.Resize(self.image_size))

        img_transform = torchvision.transforms.Compose(transform_list)
        

        stl10_full = torchvision.datasets.STL10(self.train_path,split='train',transform=img_transform)
        self.stl10_train,self.stl10_val = random_split(stl10_full,[int(0.8*stl10_full.__len__()),int(0.2*stl10_full.__len__())])

        self.stl10_test = torchvision.datasets.STL10(self.test_path,split='test',transform=img_transform)

    def train_dataloader(self):
        return DataLoader(self.stl10_train,batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.stl10_test,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.stl10_val,batch_size=self.batch_size)


class LSUNDataset(pl.LightningDataModule):
    
    def __init__(self,train_path='data/lsun/train',test_path='data/lsun/test',val_path='data/lsun/val',batch_size=64,img_size=None):
        self.train_path=train_path #path to download
        self.test_path=test_path
        self.val_path=val_path
        self.batch_size=batch_size
        self.image_size=img_size

    def prepare_data(self):
        torchvision.datasets.LSUN(self.train_path,classes='train',download=True)
        torchvision.datasets.LSUN(self.val_path,classes='val',download=True)
        torchvision.datasets.LSUN(self.test_path,classes='test',download=True)

    def setup(self):

        #apply transforms and splits
        transform_list=[torchvision.transforms.ToTensor()]
        if self.image_size is not None:
            transform_list.append(transforms.Resize(self.image_size))

        img_transform = torchvision.transforms.Compose(transform_list)

        self.lsun_train=torchvision.datasets.LSUN(self.train_path,classes='train',transform=img_transform)
        self.lsun_val=torchvision.datasets.LSUN(self.val_path,classes='val',transform=img_transform)
        self.lsun_test = torchvision.datasets.lsun(self.test_path,classes='test',transform=img_transform)

    def train_dataloader(self):
        return DataLoader(self.lsun_train,batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.lsun_test,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.lsun_val,batch_size=self.batch_size)




