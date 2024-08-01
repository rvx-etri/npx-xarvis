from pathlib import *
import math
import copy
import os

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
from npx_text_parser import *

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
  def __init__(self, net_cfg_path:Path, neuron_type_str:NpxNeuronType):
    super(NpxModule, self).__init__()

    self.net_cfg_path = net_cfg_path
    self.neuron_type = NpxNeuronType(neuron_type_str) if neuron_type_str else None
    self.train_threshold = False
    #self.reset_mechanism = 'zero'
    self.reset_mechanism = 'subtract'
    self.net_parser = NpxTextParser(self.net_cfg_path)
    self.net_parser.parsing()
    self.layer_sequence = []

    net_option = self.net_parser.section_list[0]
    layer_option_list = self.net_parser.section_list[1:]
    self.dataset = self.net_parser.find_option_value(net_option, 'dataset', 'mnist')
    self.nlayer = len(layer_option_list)
    self.gen_layer_sequence(layer_option_list)
    # print(net_option, layer_option_list)

    self.set_train_mode(self.train_threshold)

  @property
  def dataset_name(self):
      return self.dataset
      
  @property
  def num_layer(self):
      return self.nlayer

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
      
  def print_parameter(self):
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

  def write_parameter(self, path:Path):
    assert path.parent.is_dir(), path
    line_list = []
    for layer, neuron in self.layer_sequence:
      line_list.append(str(layer.weight.tolist()))
      line_list.append(str(neuron.threshold.tolist()))
    path.write_text('\n'.join(line_list))

  def gen_layer_sequence(self, layer_option_list):
    num_layer = len(layer_option_list)
    for i, layer_option in enumerate(layer_option_list):
      if i == (num_layer-1):
        neuron_output = True
      else:
        neuron_output = False
          
      if layer_option.get('section'):
        if layer_option['section'] == 'fc':
          # synapse option
          in_features = int(self.net_parser.find_option_value(layer_option, 'in_features', 1))
          out_features = int(self.net_parser.find_option_value(layer_option, 'out_features', 1))
          # print(in_features, out_features)

          layer = nn.Linear(in_features, out_features, bias=False)
          neuron = self.make_neuron(layer_option, neuron_output)
          self.add_module('layer' + str(i), layer)
          self.add_module('neuron' + str(i), neuron)
          self.layer_sequence.append((layer, neuron))
          
        elif layer_option['section'] == 'conv':
          # synapse option
          in_channels = int(self.net_parser.find_option_value(layer_option, 'in_channels', 1))
          out_channels = int(self.net_parser.find_option_value(layer_option, 'out_channels', 1))
          kernel_size = int(self.net_parser.find_option_value(layer_option, 'kernel_size', 3))
          stride = int(self.net_parser.find_option_value(layer_option, 'stride', 1))
          padding = int(self.net_parser.find_option_value(layer_option, 'padding', 0))
          # print(in_channels, out_channels, kernel_size, stride, padding)

          layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
          neuron = self.make_neuron(layer_option, neuron_output)
          self.add_module('layer' + str(i), layer)
          self.add_module('neuron' + str(i), neuron)
          self.layer_sequence.append((layer, neuron))
            
        else:
          print('unsupported layer')
            
      else:
        print('It is not layer option')

  def make_neuron(self, layer_option, neuron_output):
    beta = int(self.net_parser.find_option_value(layer_option, 'beta', 1))
    reset_mechanism = self.net_parser.find_option_value(layer_option, 'reset_mechanism', 'subtract')
    threshold = float(self.net_parser.find_option_value(layer_option, 'threshold', 1.0))
    learn_threshold = self.net_parser.find_option_value(layer_option, 'learn_threshold', False)
    neuron = snntorch.Leaky(beta=beta, threshold=threshold, init_hidden=True, 
                reset_mechanism=reset_mechanism, learn_threshold=learn_threshold, output=neuron_output)
    return neuron
