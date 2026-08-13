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
from npx_cfg_parser import *

import npx_app_cfg_generator

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

NEURON_REGISTRY = {
  'leaky': snntorch.Leaky,
  'synaptic': snntorch.Synaptic,
  'alpha': snntorch.Alpha,
}
NEURON_STATE_NAMES = {
  'leaky': ('mem',),
  'synaptic': ('syn', 'mem'),
  'alpha': ('syn_exc', 'syn_inh', 'mem'),
}
NEURON_SECTION_NAMES = ('Leaky', 'Synaptic', 'Alpha')
NEURON_CLASSES = tuple(NEURON_REGISTRY.values())

class NpxModule(nn.Module):
  def __init__(self, app_cfg_path:Path, neuron_type_class=NpxNeuronType):
    super().__init__()
    self.neuron_type_class = neuron_type_class
    self.app_cfg_path = app_cfg_path
    if self.app_cfg_path and self.app_cfg_path.is_file():
      self.cfg_parser = NpxCfgParser()
      self.cfg_parser.parse_file(self.app_cfg_path)
      
      self.layer_sequence = []
      self.gen_layer_sequence(self.cfg_parser.layer_info_list)
      # print(net_option, layer_option_list)
    self.is_network_quantized = False
  
  def global_config(self, option_name:str):
    return self.cfg_parser.global_info.get(option_name)
      
  @property
  def app_name(self):
    return self.app_cfg_path.stem
  
  @property
  def dataset_name(self):
    return self.dataset
      
  @property
  def num_layer(self):
    return len(self.cfg_parser.layer_info_list)
  
  @property
  def input_size(self):
    return self.cfg_parser.global_info['input_size']
  
  @property
  def timesteps(self):
    return self.cfg_parser.preprocess_info['timesteps']

  @classmethod
  def is_neuron(cls, layer):
    return isinstance(layer, NEURON_CLASSES)

  def backup_epoch_cfg(self, cfg_path:Path, overwrite:bool=False):
    assert overwrite or (not cfg_path.is_file()), cfg_path
    app_cfg_generator = npx_app_cfg_generator.NpxAppCfgGenerator()
    app_cfg_generator.import_module(self)
    app_cfg_generator.write_file(cfg_path)
  
  def backup_raw_cfg(self, cfg_path:Path):
    contents = self.app_cfg_path.read_text()
    if cfg_path.is_file():
      previous_contents = cfg_path.read_text()
      assert contents==previous_contents
    cfg_path.write_text(contents)

  def backup_cfg(self, npx_define:NpxDefine, epoch_index:int):
    self.backup_raw_cfg(npx_define.get_parameter_raw_cfg_path())
    self.backup_epoch_cfg(npx_define.get_parameter_epoch_cfg_path(epoch_index),True)

  def forward(self, x:Tensor):
    last_tensor = x
    last_tensor_list = []
    for i, layer in enumerate(self.layer_sequence):
      if (type(layer)==nn.Linear) or (type(layer)==nn.Conv2d):
        last_tensor = self.forward_layer(i, layer, last_tensor)
      elif isinstance(layer, Shortcut):
        skip_tensor = last_tensor_list[(i) + layer.skip_from]
        if layer.mode == "projection":
          skip_tensor = self.forward_layer(i, layer, skip_tensor)
        else: # ??
          skip_tensor = layer(skip_tensor)
        last_tensor = last_tensor + skip_tensor
      elif self.is_neuron(layer):
        last_tensor = self.forward_neuron(i, layer, last_tensor)
      else:
        last_tensor = layer(last_tensor)
      last_tensor_list.append(last_tensor)

    return last_tensor

  def forward_layer(self, i:int, layer, x:Tensor):
    if self.training and layer.neuron_type:
      original_tensor = copy.deepcopy(layer.weight.data)
      layer.neuron_type.synch_with_threshold(layer.neuron.threshold)
      layer.neuron_type.update_mapped_fvalue(layer.weight.data)
      qtensor = layer.neuron_type.quantize_tensor(layer.weight.data, bounded=True)
      layer.weight.data = layer.neuron_type.dequantize_tensor(qtensor)
    current = layer(x)
    if self.training and layer.neuron_type:
      layer.weight.data = original_tensor
      layer.neuron_type.clamp_weight_(layer.weight.data, self.is_network_quantized)
    return current
      
  def forward_neuron(self, i:int, neuron, x:Tensor):
    #if self.training and self.can_learn_neural_threshold and self.does_neuron_learn_threshold(neuron):
    current = neuron(x)
    neuron_type:NpxNeuronType = neuron.neuron_type
    if neuron_type:
      neuron_type.clamp_mem_(neuron.mem, self.is_network_quantized)
      if neuron_type.learn_beta:
        neuron.beta.data.fill_(neuron_type.quantize_beta(neuron.beta.data.float()))
    return current
      
  def print_parameter(self):
    for layer in self.layer_sequence:
      if (type(layer)==nn.Linear) or (type(layer)==nn.Conv2d):
        print(layer.weight)
      elif isinstance(layer, Shortcut):
        if layer.mode == "projection":
          print(layer.weight)
      elif self.is_neuron(layer):
        print(layer.threshold)

  def quantize_network(self):
    assert not self.training
    self.is_network_quantized = True
    for layer in self.layer_sequence:
      if (type(layer)==nn.Linear) or (type(layer)==nn.Conv2d):
        qtensor = layer.neuron_type.quantize_tensor(layer.weight.data, bounded=True)
        layer.weight.data = qtensor.tensor.float()
      elif isinstance(layer, Shortcut):
        if layer.mode == "projection":
          qtensor = layer.neuron_type.quantize_tensor(layer.weight.data, bounded=True)
          layer.weight.data = qtensor.tensor.float()
      elif self.is_neuron(layer):
        qtensor = layer.neuron_type.quantize_tensor(layer.threshold, bounded=False)
        layer.threshold = type(layer.threshold)(qtensor.tensor.float())

  def write_parameter(self, path:Path):
    assert path.parent.is_dir(), path
    line_list = []
    for layer in self.layer_sequence:
      if (type(layer)==nn.Linear) or (type(layer)==nn.Conv2d):
        line_list.append(str(layer.weight.tolist()))
      elif isinstance(layer, Shortcut):
        if layer.mode == "projection":
          line_list.append(str(layer.weight.tolist()))
      elif self.is_neuron(layer):
        line_list.append(str(layer.threshold.tolist()))
        line_list.append(str(layer.beta.tolist()))
    path.write_text('\n'.join(line_list))
  
  def dicide_option_value(self, layer_option:dict, option_name:str, default_value):
    global_value = self.global_config(option_name)
    local_value = layer_option.get(option_name)
    if local_value!=None:
      result = local_value
    elif global_value!=None:
      result = global_value
    else:
      result = default_value
    assert result!=None
    layer_option[option_name] = result
    return result

  def gen_layer_sequence(self, layer_option_list):
    #num_layer = len(layer_option_list)
    not_assigned_layer_list = []
    for i, layer_option in enumerate(layer_option_list):
      #if i == (num_layer-1):
      #  neuron_output = True
      #else:
      #  neuron_output = False
      
      if layer_option.name == 'Linear':
        # synapse option
        in_features = layer_option.setdefault('in_features', 1)
        out_features = layer_option.setdefault('out_features', 1)
        # print(in_features, out_features)

        layer = nn.Linear(in_features, out_features, bias=False)
        not_assigned_layer_list.append((layer, layer_option))
        
      elif layer_option.name == 'Conv2d':
        # synapse option
        in_channels = layer_option.setdefault('in_channels', 1)
        out_channels = layer_option.setdefault('out_channels', 1)
        kernel_size = layer_option.setdefault('kernel_size', 3)
        stride = layer_option.setdefault('stride', 1)
        padding = layer_option.setdefault('padding', 0)
        groups = layer_option.setdefault('groups', 1)
        #print(in_channels, out_channels, kernel_size, stride, padding, groups)

        layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        not_assigned_layer_list.append((layer, layer_option))

      elif layer_option.name == 'Shortcut':
        # synapse option
        skip_from = layer_option.setdefault('from', 1)
        mode = layer_option.setdefault('mode', 'projection')
        in_channels = layer_option.setdefault('in_channels', 1)
        out_channels = layer_option.setdefault('out_channels', 1)
        kernel_size = layer_option.setdefault('kernel_size', 1)
        stride = layer_option.setdefault('stride', 1)
        padding = layer_option.setdefault('padding', 0)
        # print(in_channels, out_channels, kernel_size, stride, padding)

        layer = Shortcut(in_channels, out_channels, kernel_size, stride, padding, bias=False, skip_from=skip_from, mode=mode)
        not_assigned_layer_list.append((layer, layer_option))

      elif layer_option.name == 'MaxPool2d':
        kernel_size = layer_option.setdefault('kernel_size', 1)
        stride = layer_option.setdefault('stride', kernel_size)
        padding = layer_option.setdefault('padding', 0)

        layer = nn.MaxPool2d(kernel_size, stride, padding)
        not_assigned_layer_list.append((layer, layer_option))
          
      elif layer_option.name == 'AvgPool2d':
        kernel_size = layer_option.setdefault('kernel_size', 1)
        stride = layer_option.setdefault('stride', kernel_size)
        padding = layer_option.setdefault('padding', 0)

        layer = nn.AvgPool2d(kernel_size, stride, padding)
        not_assigned_layer_list.append((layer, layer_option))

      elif layer_option.name == 'Flatten':
        layer = nn.Flatten()
        not_assigned_layer_list.append((layer, layer_option))

      elif layer_option.name in NEURON_SECTION_NAMES:
        #layer = self.make_neuron(layer_option, neuron_output)
        layer = self.make_neuron(layer_option, False)
        assert layer.neuron_type
        for previous_layer, previous_layer_option in not_assigned_layer_list:
          previous_layer.neuron = layer
          previous_layer.neuron_type = layer.neuron_type
          assert 'neuron_type' not in previous_layer_option
          previous_layer_option['neuron_type'] = layer.neuron_type.name
        not_assigned_layer_list = []
      else:
        assert 0

      self.add_module('layer' + str(i), layer)
      self.layer_sequence.append(layer)
    assert len(not_assigned_layer_list)==0

  def make_neuron(self, layer_option, neuron_output):
    neuron_type_str = self.dicide_option_value(layer_option, 'neuron_type', 'q8ssf')
    neuron_type = self.neuron_type_class(neuron_type_str)
    
    mapped_fvalue = self.dicide_option_value(layer_option, 'mapped_fvalue', neuron_type.mapped_fvalue)
    neuron_type.mapped_fvalue = mapped_fvalue
    layer_option['mapped_fvalue'] = neuron_type.mapped_fvalue
    
    beta = self.dicide_option_value(layer_option, 'beta', 1.0)
    beta = neuron_type.quantize_beta(beta)
    layer_option['beta'] = beta
    
    if neuron_type.can_learn_beta:
      learn_beta = self.dicide_option_value(layer_option, 'learn_beta', False)
    else:
      layer_option['learn_beta'] = False
    neuron_type.learn_beta = learn_beta
    
    reset_mechanism = self.dicide_option_value(layer_option, 'reset_mechanism', 'subtract')    
    reset_delay = self.dicide_option_value(layer_option, 'reset_delay', True)
    
    threshold = self.dicide_option_value(layer_option, 'threshold', 1.0)
    if neuron_type.can_learn_threshold:
      learn_threshold = self.dicide_option_value(layer_option, 'learn_threshold', False)
    else:
      layer_option['learn_threshold'] = False
      learn_threshold = False
    neuron_type.learn_threshold = learn_threshold

    # Surrogate-gradient width, tied to each layer's own threshold.
    #
    # The default (spike_grad=None) is snntorch's atan surrogate at alpha=2,
    # whose half-width 2/(pi*alpha)=0.318 is an *absolute* distance in membrane
    # units. Each layer's membrane scale tracks its own threshold, so once
    # thresholds are calibrated per layer one fixed window fits them all badly:
    # at thresholds 0.087..0.487 that same 0.318 is 3.7x the threshold in the
    # shallow layers -- gradient nearly constant, so they stop discriminating
    # near-threshold inputs -- and 0.65x in the deep ones.
    #
    # fast_sigmoid is used rather than atan because its gradient
    #   1 / (slope*|mem-threshold| + 1)**2
    # peaks at 1.0 for every slope, so changing slope changes the width and
    # nothing else. atan is normalised to unit *integral* instead, meaning its
    # peak is alpha/2 -- narrowing it there also multiplies the gradient (11x in
    # the shallow layers at alpha=2/threshold), which compounds across layers
    # and silences the network exactly as too large an lr does.
    #
    # `surrogate_scale` is the slope you would want at threshold=1.0; the slope
    # actually used is surrogate_scale/threshold, so the window stays a fixed
    # fraction of threshold in every layer. Half-width is 0.4142/slope, so
    # surrogate_scale=1.3 reproduces the default 0.318 window at threshold=1.0.
    # Unset (or 0) keeps the previous behaviour exactly, so existing apps are
    # unaffected. Training-only: it never reaches the riscv network cfg.
    surrogate_scale = self.dicide_option_value(layer_option, 'surrogate_scale', 0)
    if surrogate_scale:
      assert threshold > 0, f'surrogate_scale needs threshold>0 (got {threshold})'
      spike_grad = surrogate.fast_sigmoid(slope=float(surrogate_scale)/float(threshold))
    else:
      spike_grad = None
    neuron = snntorch.Leaky(beta=beta, learn_beta=learn_beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold,
                            init_hidden=True, reset_delay=reset_delay, reset_mechanism=reset_mechanism, output=neuron_output)
    neuron.neuron_type = neuron_type
    
    return neuron

class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, skip_from, mode="projection"):
        super().__init__()

        self.skip_from = skip_from
        self.mode = mode

        if self.mode == "projection":
            self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            self.weight = self.op.weight
        elif self.mode == "identity":
            self.op = nn.Identity()
            self.weight = None
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, x):
        return self.op(x)
