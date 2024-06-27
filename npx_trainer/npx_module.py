from pathlib import *
import math
import copy

import torch
import torch.nn as nn
import snntorch
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
from snntorch import spikegen

from npx_define import *
from npx_neuron_type import *

class PotentialResult():
  def __init__(self, pacc:int=0, nacc:int=0, neuron_type:NpxNeuronType=None):
    self.neuron_type = neuron_type
    self.pacc = pacc
    self.nacc = nacc

  def update(self, result):
    self.pacc = max(self.pacc,result.pacc)
    self.nacc = max(self.nacc,result.nacc)

  @property
  def max(self):
    return max(self.pacc,self.nacc)

  def __repr__(self):
    assert self.neuron_type
    return str((self.pacc,self.nacc,math.ceil(self.max/self.neuron_type.qmax)))

class NpxModule(nn.Module):
  def __init__(self, app_name:str, neuron_type_str:NpxNeuronType):
    super(NpxModule, self).__init__()

    self.app_name = app_name
    self.neuron_type = NpxNeuronType(neuron_type_str) if neuron_type_str else None
    self.train_threshold = False
    #self.reset_mechanism = 'zero'
    self.reset_mechanism = 'subtract'

    #if self.neuron_type and self.neuron_type.is_infinite_potential:
    #    conv_lif_threshold = 1.25
    #    fc_lif_threshold = 1.0
    beta = 1
    conv_lif_threshold = 1.0
    fc_lif_threshold = 1.0

    self.layer_sequence = []
    
    if self.app_name=='mnist_l1f':
      self.layer1 = nn.Linear(14*14, 10, bias=False)
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer1,self.neuron1))
                    
    elif self.app_name=='mnist_l2cf': # 256x64
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
        
      self.layer2= nn.Linear(10*10*2, 10, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)            
      self.layer_sequence.append((self.layer2,self.neuron2))

    elif self.app_name=='mnist_l2ff': # 256x64
      self.layer1 = nn.Linear(14*14, 256, bias=False)
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
        
      self.layer2 = nn.Linear(256, 10, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer2,self.neuron2))

    elif self.app_name=='mnist_l3fff': # 256x64
      self.layer1 = nn.Linear(14*14, 256, bias=False)
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
        
      self.layer2 = nn.Linear(256, 256, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))

      self.layer3 = nn.Linear(256, 10, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer3,self.neuron3))

    elif self.app_name=='mnist_l3cff': # 256x64
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
        
      self.layer2 = nn.Linear(10*10*2, 256, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))

      self.layer3 = nn.Linear(256, 10, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer3,self.neuron3))

    elif self.app_name=='mnist_l3ccf': # 256x64
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
        
      self.layer2 = nn.Conv2d(2, 7, filter_size, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))

      self.layer3 = nn.Linear(6*6*7, 10, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer3,self.neuron3))

    elif self.app_name=='fmnist_l2cf':
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))

      self.layer2= nn.Linear(10*10*2, 10, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer2,self.neuron2))

    elif self.app_name=='fmnist_l3cff':
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
        
      self.layer2 = nn.Linear(10*10*2, 256, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))

      self.layer3 = nn.Linear(256, 10, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer3,self.neuron3))

    elif self.app_name=='fmnist_l3ccf':
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
        
      self.layer2 = nn.Conv2d(2, 7, filter_size, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))

      self.layer3 = nn.Linear(6*6*7, 10, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer3,self.neuron3))

    elif self.app_name=='cifar10_l5cccff': 
      filter_size = 3
      self.layer1 = nn.Conv2d(3, 8, filter_size, bias=False) # 30x30x8
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
        
      self.layer2 = nn.Conv2d(8, 16, filter_size, bias=False) # 28x28x16
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))

      self.layer3 = nn.Conv2d(16, 32, filter_size, bias=False) # 26x26x32
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer3,self.neuron3))

      self.layer4 = nn.Linear(26*26*32, 256, bias=False)
      self.neuron4 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer4,self.neuron4))

      self.layer5 = nn.Linear(256, 10, bias=False)
      self.neuron5 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer5,self.neuron5))

    elif self.app_name=='gtsrb_l5cccff': 
      filter_size = 5
      self.layer1 = nn.Conv2d(3, 10, filter_size, bias=False) # 28x28x10 
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
        
      self.layer2 = nn.Conv2d(10, 15, filter_size, bias=False) # 24x24x15
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))

      self.layer3 = nn.Conv2d(15, 25, filter_size, bias=False) # 20x20x25
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer3,self.neuron3))

      self.layer4 = nn.Linear(20*20*25, 350, bias=False)
      self.neuron4 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer4,self.neuron4))

      self.layer5 = nn.Linear(350, 43, bias=False)
      self.neuron5 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer5,self.neuron5))

    else:
      assert 0, self.app_name

    self.set_train_mode(self.train_threshold)
        
    '''
    elif self.app_name=='mnist_l3ccf':
      num_hidden1 = 5
      num_hidden1 = 12

      self.conv1 = nn.Conv2d(1, num_hidden1, filter_size, bias=False)
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, learn_threshold=self.can_learn_threshold)
        
      self.conv2 = nn.Conv2d(num_hidden1, num_hidden2, filter_size, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, learn_threshold=self.can_learn_threshold)
      
      self.fc1 = nn.Linear(6*6*num_hidden2, 10, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, learn_threshold=self.can_learn_threshold, output=True)

    elif self.app_name=='cifar-10_5l':
      self.conv1 = nn.Conv2d(3, 6, filter_size, bias=False) # 16x16x3 -> 12x12x6
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, learn_threshold=self.can_learn_threshold)
      
      self.conv2 = nn.Conv2d(6, 16, filter_size, bias=False) # 12x12x6 -> 8x8x16
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, learn_threshold=self.can_learn_threshold)
      
      self.fc1 = nn.Linear(8*8*16, 120, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, learn_threshold=self.can_learn_threshold)
  
      self.layer2 = nn.Linear(120, 84, bias=False)
      self.neuron4 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, learn_threshold=self.can_learn_threshold)
  
      self.layer3 = nn.Linear(84, 10, bias=False)
      self.neuron5 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, learn_threshold=self.can_learn_threshold, output=True)
    '''
    
  @property
  def dataset_name(self):
    return self.app_name.split('_')[0]
      
  @property
  def num_layer(self):
    return int(self.app_name.split('_')[1][1])

  @property
  def can_learn_threshold(self):
    return False
    #return True if self.neuron_type and self.neuron_type.is_infinite_potential else False
  
  def set_train_mode(self, train_threshold:bool=False):
    pass
    '''
    self.train_threshold = train_threshold
    if self.can_learn_threshold:
      for layer, neuron in self.layer_sequence:
        if self.train_threshold:
          layer.weight.requires_grad_(False)
          neuron.threshold.requires_grad_(True)
        else:
          layer.weight.requires_grad_(True)
          neuron.threshold.requires_grad_(False)
    '''

  def forward(self, x:Tensor):
    last_tensor = x
    for i, (layer, neuron) in enumerate(self.layer_sequence):
      if type(layer)==nn.Linear:
        last_tensor = torch.flatten(last_tensor, 1)
      current = self.forward_layer(layer, last_tensor)
      last_tensor = self.forward_neuron(neuron, current)
    return last_tensor

  def forward_layer(self, layer, x:Tensor):
    if self.training and self.neuron_type:
      original_tensor = copy.deepcopy(layer.weight.data)
      qtensor = self.neuron_type.quantize_tensor(layer.weight.data)
      layer.weight.data = self.neuron_type.dequantize_tensor(qtensor)
    current = layer(x)
    if self.training and self.neuron_type:
      if not self.neuron_type.is_signed_weight:
        original_tensor = original_tensor.clamp(min=0)
      layer.weight.data = original_tensor
    return current
      
  def forward_neuron(self, neuron, x:Tensor):
    if self.training and self.train_threshold and self.can_learn_threshold:
      qtensor = self.neuron_type.quantize_tensor(neuron.threshold, bounded=False)
      neuron.threshold = type(neuron.threshold)(self.neuron_type.dequantize_tensor(qtensor))
    current = neuron(x)
    if self.neuron_type:
      if not self.neuron_type.is_signed_potential:
        neuron.mem = neuron.mem.clamp(min=0)
    return current
      
  def print_cfg(self):
    for layer, neuron in self.layer_sequence:
      print(layer.weight)
      print(neuron.threshold)

  def quantize_network(self):
    assert not self.training
    assert self.neuron_type
    for layer, neuron in self.layer_sequence:
      qtensor = self.neuron_type.quantize_tensor(layer.weight.data)
      layer.weight.data = qtensor.tensor.float()
      quantized_threshold = (neuron.threshold / qtensor.scale).round()
      neuron.threshold = type(neuron.threshold)(quantized_threshold)

  def write_cfg(self, path:Path):
    assert path.parent.is_dir(), path
    line_list = []
    for layer, neuron in self.layer_sequence:
      line_list.append(str(layer.weight.tolist()))
      line_list.append(str(neuron.threshold.tolist()))
    path.write_text('\n'.join(line_list))