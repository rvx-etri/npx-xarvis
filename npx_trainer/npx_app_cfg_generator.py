from pathlib import *
import math
import copy
import os
import argparse

import torch
import torch.nn as nn
import snntorch

from collections import Counter

from npx_define import *
from npx_neuron_type import *
from npx_data_manager import *

class NpxAppCfgGenerator(nn.Module):
  def __init__(self, app_name:str, neuron_type_str:str, dataset_path:Path, app_cfg_dir_path:Path):
    super().__init__()

    self.app_name = app_name
    self.neuron_type = NpxNeuronType(neuron_type_str) if neuron_type_str else None
    self.train_threshold = False
    self.reset_mechanism = 'zero'
    #self.reset_mechanism = 'subtract'

    # beta = 1
    # conv_lif_threshold = 1.0
    # fc_lif_threshold = 1.0

    npx_data_manager = NpxDataManager(dataset_name=self.dataset_name, dataset_path=dataset_path, num_kfold=5)
    input_size = npx_data_manager.dataset_test[0][0].shape
    classes = len(dict(Counter(sample_tup[1] for sample_tup in npx_data_manager.dataset_test)))

    # print(self.app_cfg_path)
    self.net_parser = NpxTextParser(self.app_cfg_path)

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
      self.gen_fc_section(in_features=14*14, out_features=10, 
                     input_type='value', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='mnist_l2cf': # 256x64
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=10*10*2, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='mnist_l2ff': # 256x64
      self.gen_fc_section(in_features=14*14, out_features=256, 
                     input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='mnist_l3fff': # 256x64
      self.gen_fc_section(in_features=14*14, out_features=256, 
                     input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=256, out_features=256, 
                     input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='mnist_l3cff': # 256x64
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=10*10*2, out_features=256, 
                     input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='mnist_l3ccf': # 256x64
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_conv_section(in_channels=2, out_channels=7, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=6*6*7, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='fmnist_l2cf':
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=10*10*2, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='fmnist_l3cff':
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=10*10*2, out_features=256, 
                     input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='fmnist_l3ccf':
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_conv_section(in_channels=2, out_channels=7, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=6*6*7, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='cifar10_l5cccff': 
      filter_size = 3
      self.gen_conv_section(in_channels=3, out_channels=8, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_conv_section(in_channels=8, out_channels=16, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_conv_section(in_channels=16, out_channels=32, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=26*26*32, out_features=256, 
                     input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    elif self.app_name=='gtsrb_l5cccff': 
      filter_size = 5
      self.gen_conv_section(in_channels=3, out_channels=10, kernel_size=filter_size,
                      input_type='value', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_conv_section(in_channels=10, out_channels=15, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_conv_section(in_channels=15, out_channels=25, kernel_size=filter_size,
                      input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=20*20*25, out_features=350, 
                     input_type='spike', output_type='spike', reset_mechanism=self.reset_mechanism)
      self.gen_fc_section(in_features=350, out_features=43, 
                     input_type='spike', output_type='value', reset_mechanism=self.reset_mechanism)

    else:
      assert 0, self.app_name
    
  @property
  def dataset_name(self):
    return self.app_name.split('_')[0]
      
  @property
  def num_layer(self):
    return int(self.app_name.split('_')[1][1])
  
  @property
  def app_cfg_path(self):
    return app_cfg_dir_path / f'{self.app_name}.cfg'

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

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='NPX Framework')
  parser.add_argument('-app', '-a', nargs='+', help='example app name', default=[])
  parser.add_argument('-neuron', '-n', nargs='+', help='types of neuron', default=['q8ssf'])
  parser.add_argument('-dataset', '-d', help='dataset directory')
  parser.add_argument('-cfg', '-c', nargs='+', help='app cfg file name', default=[])
  parser.add_argument('-output', '-o', help='app cfg directory', default='./generated_cfg')

  # check args
  args = parser.parse_args()
  assert args.app
  assert args.dataset
  assert args.output

  exam_app_name_list = args.app

  neuron_list = []
  for neuron_set in args.neuron:
    if '-' in neuron_set:
      train_neuron_str, test_neuron_str = neuron_set.split('-')
    else:
      train_neuron_str = neuron_set
      test_neuron_str = neuron_set
    neuron_list.append((train_neuron_str,test_neuron_str))
  dataset_path = Path(args.dataset).absolute()
  if not dataset_path.is_dir():
    dataset_path.mkdir(parents=True)
  app_cfg_dir_path = Path(args.output).absolute()
  if not app_cfg_dir_path.is_dir():
    app_cfg_dir_path.mkdir(parents=True)

  #print(exam_app_name_list)
  #print(neuron_list)
  for app_name in exam_app_name_list:
    for train_neuron_str, test_neuron_str in neuron_list:
      app_cfg_generator = NpxAppCfgGenerator(app_name=app_name, neuron_type_str=train_neuron_str,
                                dataset_path=dataset_path, app_cfg_dir_path=app_cfg_dir_path)
      app_cfg_generator.net_parser.print()
      app_cfg_generator.net_parser.save()
