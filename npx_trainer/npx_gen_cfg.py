import os
import argparse
import time
import shutil
from pathlib import *
from tqdm.auto import tqdm
from collections import namedtuple
from collections import Counter

from npx_define import *
from npx_data_manager import *
from npx_module import *

class NpxGenCfg():
  def __init__(self):
    self.num_steps_to_train = 32

  def get_num_classes(self, npx_data_manager:NpxDataManager):
    class_counts = dict(Counter(sample_tup[1] for sample_tup in npx_data_manager.dataset_test))
    return len(class_counts)

  def get_cfg_text_net(self, npx_define:NpxDefine, npx_data_manager:NpxDataManager):
    #print(npx_data_manager.dataset_test)
    #print(len(npx_data_manager.dataset_test))
    #print(len(npx_data_manager.dataset_test[0]))
    #print(npx_data_manager.dataset_test[0][0].shape)
    height = npx_data_manager.dataset_test[0][0].shape[1]
    width = npx_data_manager.dataset_test[0][0].shape[2]
    channels = npx_data_manager.dataset_test[0][0].shape[0]
    classes = self.get_num_classes(npx_data_manager=npx_data_manager)
    #print(height,width,channels,classes)

    #print(npx_define.train_neuron_str)
    net_text = '[network]\n'
    net_text += 'app_name=%s\n' % npx_define.app_name
    net_text += 'neuron_type=%s\n' % npx_define.train_neuron_str
    net_text += 'height=%d\n' % height
    net_text += 'width=%d\n' % width
    net_text += 'channels=%d\n' % channels
    net_text += 'timesteps=%d\n' % self.num_steps_to_train
    net_text += 'classes=%d\n' % classes
    net_text += '\n'
    print(net_text)

    return net_text

  def get_cfg_text_layer(self, layer):
    if type(layer)==nn.Linear:
      layer_text = '[fc]\n'
      layer_text += f'in_features={layer.in_features}\n'
      layer_text += f'out_features={layer.out_features}\n'
      layer_text += f'input_type=spike\n'
      layer_text += f'output_type=spike\n'
      layer_text += '\n'
    elif type(layer)==nn.Conv2d:
      layer_text = '[conv]\n'
      layer_text += f'out_channels={layer.out_channels}\n'
      layer_text += f'kernel_size={layer.kernel_size[0]}\n'
      layer_text += f'stride={layer.stride[0]}\n'
      layer_text += f'padding={layer.padding[0]}\n'
      layer_text += f'input_type=spike\n'
      layer_text += f'output_type=spike\n'
      layer_text += '\n'
    else:
      print('unsupported layer', type(layer))

    print(layer_text)
    return layer_text

  def gen_app_cfg(self, npx_define:NpxDefine, npx_data_manager:NpxDataManager):
    npx_module = NpxModule(app_name=npx_define.app_name, neuron_type_str=npx_define.train_neuron_str)
    app_cfg_filename = f'{npx_define.app_name}_{npx_define.train_neuron_str}.cfg'
    app_cfg_path = npx_define.app_dir_path / app_cfg_filename
    #print(app_cfg_path)
    if not npx_define.app_dir_path.is_dir():
      npx_define.app_dir_path.relative_to(Path('.').absolute())
      npx_define.app_dir_path.mkdir(parents=True)

    with open(app_cfg_path, 'w') as f:
      text = self.get_cfg_text_net(npx_define, npx_data_manager)
      f.write(text)
      for i, (layer, neuron) in enumerate(npx_module.layer_sequence):
        text = self.get_cfg_text_layer(layer)
        f.write(text)

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='NPX Framework')
  parser.add_argument('-app', '-a', nargs='+', help='app name')
  parser.add_argument('-cfg', '-c', nargs='+', help='cfg file name')
  parser.add_argument('-neuron', '-n', nargs='+', help='types of neuron')
  parser.add_argument('-kfold', '-k', help='number of k-fold')
  parser.add_argument('-dataset', '-d', help='dataset directory')
  parser.add_argument('-output', '-o', help='output directory')

  # check args
  args = parser.parse_args()
  assert args.app or args.cfg
  assert args.neuron
  assert args.output

  app_name_list = args.app
  neuron_list = []
  for neuron_set in args.neuron:
    if '-' in neuron_set:
      train_neuron_str, test_neuron_str = neuron_set.split('-')
    else:
      train_neuron_str = neuron_set
      test_neuron_str = neuron_set
    neuron_list.append((train_neuron_str,test_neuron_str))
  num_kfold = int(args.kfold) if args.kfold else 5
  output_path = Path(args.output).absolute()
  if not output_path.is_dir():
    output_path.relative_to(Path('.').absolute())
    output_path.mkdir(parents=True)
  dataset_path = Path(args.dataset).absolute() if args.dataset else (output_path / 'dataset')

  # common env
  npx_gen_cfg = NpxGenCfg()

  cfg_list  = args.cfg
  
  print(app_name_list)
  print(cfg_list)
  if cfg_list != None:
    for cfg in cfg_list:
      npx_module = NpxModule(app_name=cfg, neuron_type_str=train_neuron_str)

  if app_name_list != None:
    for app_name in app_name_list:
      for train_neuron_str, test_neuron_str in neuron_list:    
        npx_define = NpxDefine(app_name=app_name, train_neuron_str=train_neuron_str, test_neuron_str=test_neuron_str, output_path=output_path)
        npx_data_manager = NpxDataManager(dataset_name=npx_define.dataset_name, dataset_path=dataset_path, num_kfold=num_kfold)
        npx_gen_cfg.gen_app_cfg(npx_define=npx_define, npx_data_manager=npx_data_manager)
