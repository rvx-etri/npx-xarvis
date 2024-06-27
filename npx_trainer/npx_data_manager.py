from pathlib import *

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

class NpxDataManager():
  def __init__(self, dataset_name:str, dataset_path:Path, num_kfold:int=None):
    self.name = dataset_name
    self.download_path = dataset_path / self.name
    if num_kfold==None:
      num_kfold = 5
    assert num_kfold>=4, num_kfold
    self.num_kfold = num_kfold
    self.fair_distribution = False

    self.download_path.mkdir(parents=True, exist_ok=True)
    
    if self.name=='mnist':
      transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])
      dataset_train_and_val = datasets.MNIST(root=self.download_path, train=True, download=True, transform=transform)
      self.dataset_test = datasets.MNIST(root=self.download_path, train=False, download=True, transform=transform)
    elif self.name=='fmnist':
      transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])
      dataset_train_and_val = datasets.FashionMNIST(root=self.download_path, train=True, download=True, transform=transform)
      self.dataset_test = datasets.FashionMNIST(root=self.download_path, train=False, download=True, transform=transform)
    elif self.name=='cifar10':
      transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))])
      dataset_train_and_val = datasets.CIFAR10(root=self.download_path, train=True, download=True, transform=transform)
      self.dataset_test = datasets.CIFAR10(root=self.download_path, train=False, download=True, transform=transform)
    elif self.name=='gtsrb':
      transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))])
      dataset_train_and_val = datasets.GTSRB(root=self.download_path, split='train', download=True, transform=transform)
      self.dataset_test = datasets.GTSRB(root=self.download_path, split='test', download=True, transform=transform)
    else:
      assert 0, dataset_name
    
    if self.fair_distribution:
      labeled_dataset_dict = {}
      for data, label in dataset_train_and_val:
        labeled_dataset = labeled_dataset_dict.get(label)
        if labeled_dataset:
          labeled_dataset.append((data, label))
        else:
          labeled_dataset_dict[label] = [(data, label)]
      self.dataset_both_list = None
      for labeled_dataset in labeled_dataset_dict.values():
        chunk_size = int(len(labeled_dataset)/self.num_kfold)
        chunk_size_list = []
        for i in range(0, num_kfold-1):
          chunk_size_list.append(chunk_size)
        last_chunk_size = len(labeled_dataset) - (chunk_size*(num_kfold-1))
        chunk_size_list.append(last_chunk_size)
        splited_labeled_dataset = torch.utils.data.random_split(labeled_dataset, chunk_size_list)
        if not self.dataset_both_list:
          self.dataset_both_list = splited_labeled_dataset
        else:
          for i in range(len(self.dataset_both_list)):
            self.dataset_both_list[i] += splited_labeled_dataset[i]
    else:
      chunk_size = int(len(dataset_train_and_val)/self.num_kfold)
      chunk_size_list = []
      for i in range(0, num_kfold-1):
        chunk_size_list.append(chunk_size)
      last_chunk_size = len(dataset_train_and_val) - (chunk_size*(num_kfold-1))
      chunk_size_list.append(last_chunk_size)
      self.dataset_both_list = torch.utils.data.random_split(dataset_train_and_val, chunk_size_list)

  def setup_loader(self, kfold_index:int, batch_size:int=100):
    assert kfold_index <= len(self.dataset_both_list), len(self.dataset_both_list)
    self.dataset_val = self.dataset_both_list[kfold_index]
    for i, dataset_chunk in enumerate(self.dataset_both_list):
      if i==0:
        self.dataset_train = dataset_chunk
      elif i==kfold_index:
        pass
      else:
        self.dataset_train += dataset_chunk
    self.loader_list = (DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, drop_last=True),
                        DataLoader(self.dataset_val, batch_size=batch_size, shuffle=True, drop_last=True),
                        DataLoader(self.dataset_test, batch_size=batch_size, shuffle=True, drop_last=True))

  @property
  def train_loader(self):
    return self.loader_list[0]
  @property
  def val_loader(self):
    return self.loader_list[1]
  @property
  def test_loader(self):
    return self.loader_list[2]
