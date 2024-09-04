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
    
  # self.training is changed by train() and eval()

  @property
  def max(self):
    return max(self.pacc,self.nacc)

  def __repr__(self):
    assert self.neuron_type
    return str((self.pacc,self.nacc,math.ceil(self.max/self.neuron_type.qscale)))

class NpxModule(nn.Module):
  def __init__(self, app_cfg_path:Path, neuron_type_str:str='q8ssf', neuron_type_class=NpxNeuronType):
    super().__init__()
    self.neuron_type_class = neuron_type_class

    self.neuron_type = self.neuron_type_class(neuron_type_str) if neuron_type_str else None
    self.app_cfg_path = app_cfg_path
    if self.app_cfg_path and self.app_cfg_path.is_file():
      self.network_parser = NpxTextParser(self.app_cfg_path)
      self.network_parser.parsing()
            
      info_list = (
        ('dataset', 'mnist'),('timesteps', 32)
        )
      for var_name, default_value in info_list:
        value = NpxTextParser.find_option_value(self.network_parser.global_info, var_name, default_value)
        self.__dict__[var_name] = value
      
      self.layer_sequence = []
      self.nlayer = len(self.network_parser.layer_info_list)
      self.gen_layer_sequence(self.network_parser.layer_info_list)
      # print(net_option, layer_option_list)
      
  @property
  def dataset_name(self):
    return self.dataset
      
  @property
  def num_layer(self):
    return self.nlayer

  @property
  def can_learn_neuron_threshold(self):
    return False
    #return True if self.neuron_type and self.neuron_type.is_infinite_potential else False
    
  def generate_cfg(self, cfg_path:Path):
    write_cfg = True
    current_contents = self.app_cfg_path.read_text()
    if cfg_path.is_file():
      previous_contents = cfg_path.read_text()
      if current_contents!=previous_contents:
        print(f'[WARNING] {cfg_path} is overwritten due to mismatch')
      else:
        write_cfg = False
    if write_cfg:
      cfg_path.write_text(current_contents)

  def forward(self, x:Tensor):
    last_tensor = x
    for i, (layer, neuron) in enumerate(self.layer_sequence):
      if type(layer)==nn.Linear:
        last_tensor = torch.flatten(last_tensor, 1)
      current = self.forward_layer(i, layer, last_tensor)
      last_tensor = self.forward_neuron(i, neuron, current)

    return last_tensor

  def forward_layer(self, i:int, layer, x:Tensor):
    if self.training and self.neuron_type:
      original_tensor = copy.deepcopy(layer.weight.data)
      qtensor = self.neuron_type.quantize_tensor(layer.weight.data, bounded=True)
      layer.weight.data = self.neuron_type.dequantize_tensor(qtensor)
    current = layer(x)
    if self.training and self.neuron_type:
      if not self.neuron_type.is_signed_weight:
        original_tensor = original_tensor.clamp(min=0)
      layer.weight.data = original_tensor
    return current
      
  def forward_neuron(self, i:int, neuron, x:Tensor):
    if self.training and self.can_learn_neuron_threshold and neuron.learn_threshold:
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
      qtensor = self.neuron_type.quantize_tensor(layer.weight.data, bounded=True)
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
          in_features = int(NpxTextParser.find_option_value(layer_option, 'in_features', 1))
          out_features = int(NpxTextParser.find_option_value(layer_option, 'out_features', 1))
          # print(in_features, out_features)

          layer = nn.Linear(in_features, out_features, bias=False)
          neuron = self.make_neuron(layer_option, neuron_output)
          self.add_module('layer' + str(i), layer)
          self.add_module('neuron' + str(i), neuron)
          self.layer_sequence.append((layer, neuron))
          
        elif layer_option['section'] == 'conv':
          # synapse option
          in_channels = int(NpxTextParser.find_option_value(layer_option, 'in_channels', 1))
          out_channels = int(NpxTextParser.find_option_value(layer_option, 'out_channels', 1))
          kernel_size = int(NpxTextParser.find_option_value(layer_option, 'kernel_size', 3))
          stride = int(NpxTextParser.find_option_value(layer_option, 'stride', 1))
          padding = int(NpxTextParser.find_option_value(layer_option, 'padding', 0))
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
    beta = NpxTextParser.find_option_value(layer_option, 'beta', 1.0)
    reset_mechanism = NpxTextParser.find_option_value(layer_option, 'reset_mechanism', 'zero')
    threshold = float(NpxTextParser.find_option_value(layer_option, 'threshold', 1.0))
    learn_threshold = NpxTextParser.find_option_value(layer_option, 'learn_threshold', False)
    neuron = snntorch.Leaky(beta=beta, threshold=threshold, init_hidden=True, 
                reset_mechanism=reset_mechanism, learn_threshold=learn_threshold, output=neuron_output)
    return neuron
