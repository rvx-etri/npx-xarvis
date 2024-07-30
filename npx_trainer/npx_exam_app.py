from pathlib import *
import math
import copy
import os

import torch
import torch.nn as nn
import snntorch

from collections import Counter

from npx_define import *
from npx_neuron_type import *
from npx_data_manager import *

class NpxExamApp(nn.Module):
  def __init__(self, app_name:str, neuron_type_str:str, dataset_path:Path, net_cfg_dir_path:Path):
    super(NpxExamApp, self).__init__()

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


    file_name = app_name + '.cfg'
    self.net_cfg_path = net_cfg_dir_path / file_name

    npx_data_manager = NpxDataManager(dataset_name=self.dataset_name, dataset_path=dataset_path, num_kfold=5)
    input_size = npx_data_manager.dataset_test[0][0].shape
    classes = self.get_num_classes(npx_data_manager=npx_data_manager)

    # print(self.net_cfg_path)
    self.net_parser = NpxTextParser(self.net_cfg_path)
    self.layer_sequence = []

    self.net_parser.add_section('network')
    self.net_parser.add_option(-1, 'dataset', self.dataset_name)
    self.net_parser.add_option(-1, 'neuron_type', neuron_type_str)
    self.net_parser.add_option(-1, 'height', str(input_size[1]))
    self.net_parser.add_option(-1, 'weigth', str(input_size[2]))
    self.net_parser.add_option(-1, 'channels', str(input_size[0]))
    self.net_parser.add_option(-1, 'timesteps', str(32))
    self.net_parser.add_option(-1, 'classes', str(classes))

    # print(self.net_parser.section_list)

    if self.app_name=='mnist_l1f':
      self.layer1 = nn.Linear(14*14, 10, bias=False)
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer1,self.neuron1))
      self.gen_fc_section(in_features=14*14, out_features=10, 
                     input_type='value', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='mnist_l2cf': # 256x64
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer2= nn.Linear(10*10*2, 10, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)            
      self.layer_sequence.append((self.layer2,self.neuron2))
      self.gen_fc_section(in_features=10*10*2, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='mnist_l2ff': # 256x64
      self.layer1 = nn.Linear(14*14, 256, bias=False)
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
      self.gen_fc_section(in_features=14*14, out_features=256, 
                     input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer2 = nn.Linear(256, 10, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer2,self.neuron2))
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='mnist_l3fff': # 256x64
      self.layer1 = nn.Linear(14*14, 256, bias=False)
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
      self.gen_fc_section(in_features=14*14, out_features=256, 
                     input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer2 = nn.Linear(256, 256, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))
      self.gen_fc_section(in_features=256, out_features=256, 
                     input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer3 = nn.Linear(256, 10, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer3,self.neuron3))
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='mnist_l3cff': # 256x64
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer2 = nn.Linear(10*10*2, 256, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))
      self.gen_fc_section(in_features=10*10*2, out_features=256, 
                     input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer3 = nn.Linear(256, 10, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer3,self.neuron3))
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='mnist_l3ccf': # 256x64
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer2 = nn.Conv2d(2, 7, filter_size, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))
      self.gen_conv_section(in_channels=2, out_channels=7, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer3 = nn.Linear(6*6*7, 10, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer3,self.neuron3))
      self.gen_fc_section(in_features=6*6*7, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='fmnist_l2cf':
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer2= nn.Linear(10*10*2, 10, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer2,self.neuron2))
      self.gen_fc_section(in_features=10*10*2, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='fmnist_l3cff':
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer2 = nn.Linear(10*10*2, 256, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))
      self.gen_fc_section(in_features=10*10*2, out_features=256, 
                     input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer3 = nn.Linear(256, 10, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer3,self.neuron3))
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='fmnist_l3ccf':
      filter_size = 5
      self.layer1 = nn.Conv2d(1, 2, filter_size, bias=False) # 10x10x2
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer2 = nn.Conv2d(2, 7, filter_size, bias=False)
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))
      self.gen_conv_section(in_channels=2, out_channels=7, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer3 = nn.Linear(6*6*7, 10, bias=False)
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer3,self.neuron3))
      self.gen_fc_section(in_features=6*6*7, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='cifar10_l5cccff': 
      filter_size = 3
      self.layer1 = nn.Conv2d(3, 8, filter_size, bias=False) # 30x30x8
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
      self.gen_conv_section(in_channels=3, out_channels=8, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer2 = nn.Conv2d(8, 16, filter_size, bias=False) # 28x28x16
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))
      self.gen_conv_section(in_channels=8, out_channels=16, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer3 = nn.Conv2d(16, 32, filter_size, bias=False) # 26x26x32
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer3,self.neuron3))
      self.gen_conv_section(in_channels=16, out_channels=32, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer4 = nn.Linear(26*26*32, 256, bias=False)
      self.neuron4 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer4,self.neuron4))
      self.gen_fc_section(in_features=26*26*32, out_features=256, 
                     input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer5 = nn.Linear(256, 10, bias=False)
      self.neuron5 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer5,self.neuron5))
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='gtsrb_l5cccff': 
      filter_size = 5
      self.layer1 = nn.Conv2d(3, 10, filter_size, bias=False) # 28x28x10 
      self.neuron1 = snntorch.Leaky(beta=beta, threshold=conv_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer1,self.neuron1))
      self.gen_conv_section(in_channels=3, out_channels=10, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer2 = nn.Conv2d(10, 15, filter_size, bias=False) # 24x24x15
      self.neuron2 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer2,self.neuron2))
      self.gen_conv_section(in_channels=10, out_channels=15, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer3 = nn.Conv2d(15, 25, filter_size, bias=False) # 20x20x25
      self.neuron3 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer3,self.neuron3))
      self.gen_conv_section(in_channels=15, out_channels=25, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer4 = nn.Linear(20*20*25, 350, bias=False)
      self.neuron4 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold)
      self.layer_sequence.append((self.layer4,self.neuron4))
      self.gen_fc_section(in_features=20*20*25, out_features=350, 
                     input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)

      self.layer5 = nn.Linear(350, 43, bias=False)
      self.neuron5 = snntorch.Leaky(beta=beta, threshold=fc_lif_threshold, init_hidden=True, reset_mechanism=self.reset_mechanism, learn_threshold=self.can_learn_threshold, output=True)
      self.layer_sequence.append((self.layer5,self.neuron5))
      self.gen_fc_section(in_features=350, out_features=43, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

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

  def get_num_classes(self, npx_data_manager:NpxDataManager):
    class_counts = dict(Counter(sample_tup[1] for sample_tup in npx_data_manager.dataset_test))
    return len(class_counts)

  def gen_fc_section(self, in_features:int, out_features:int, 
                     input_type='spike', output_type='spike', reset_mechanism='zero'):
    self.net_parser.add_section('fc')
    self.net_parser.add_option(-1, 'in_features', str(in_features))
    self.net_parser.add_option(-1, 'out_features', str(out_features))
    self.net_parser.add_option(-1, 'input_type', input_type)
    self.net_parser.add_option(-1, 'output_type', output_type)
    self.net_parser.add_option(-1, 'reset_mechanism', reset_mechanism)

  def gen_conv_section(self, in_channels:int, out_channels:int, 
                      kernel_size=3, stride=1, padding=0,
                      input_type='spike', output_type='spike', reset_mechanism='zero'):
    self.net_parser.add_section('conv')
    self.net_parser.add_option(-1, 'in_channels', str(in_channels))
    self.net_parser.add_option(-1, 'out_channels', str(out_channels))
    self.net_parser.add_option(-1, 'kernel_size', str(kernel_size))
    self.net_parser.add_option(-1, 'stride', str(stride))
    self.net_parser.add_option(-1, 'padding', str(padding))
    self.net_parser.add_option(-1, 'input_type', input_type)
    self.net_parser.add_option(-1, 'output_type', output_type)
    self.net_parser.add_option(-1, 'reset_mechanism', reset_mechanism)