import os
import argparse
import time
import shutil
import random
import copy

import numpy as np
from torch.utils.data.dataloader import DataLoader
from pathlib import *
from tqdm.auto import tqdm
from collections import namedtuple

from npx_define import *
from npx_data_manager import *
from npx_module import *
from npx_converter import write_data_aligned_by_4bytes

'''
def save_sample(path:Path, data:np.ndarray, is_spike:int, byte_per_data:int,
    num_steps, target:np.ndarray):
  header = np.zeros(4).astype(np.uint8)
  header[0] = is_spike
  header[1] = byte_per_data
  header[2] = num_steps
  header[3] = target[0]

  print(path)
  #print(data.shape)
  height = np.array(data.shape[2]).astype(np.int32)
  width = np.array(data.shape[3]).astype(np.int32)
  channel = np.array(data.shape[1]).astype(np.int32)
  print(height, width, channel)
  #print(type(height))

  with open(path, "wb") as data_file:
    data_file.write(header)
    data_file.write(height)
    data_file.write(width)
    data_file.write(channel)
    data_file.write(data)
  #print(type(data))
'''

def get_sample(npx_data_manager:NpxDataManager, num_sample:int):
  data_loader = DataLoader(npx_data_manager.dataset_test, batch_size=1, shuffle=True, drop_last=False)
  sample_list = []
  for i in range(num_sample):
    data, target = next(iter(data_loader))
    sample_list.append((data, target))
    #print(data, target)

  return sample_list 

def generate_testvector(npx_define:NpxDefine, npx_data_manager:NpxDataManager, num_sample:int):
  print('\n[TEST VECTOR]', npx_define.app_name)

  npx_module = NpxModule(app_cfg_path=npx_define.app_cfg_path, 
                         neuron_type_str=npx_define.train_neuron_str).to(device)
  npx_module.eval()
  riscv_parameter_path = npx_define.get_riscv_parameter_path(is_quantized=True)
  assert riscv_parameter_path.exists(), riscv_parameter_path
  npx_module.load_state_dict(torch.load(riscv_parameter_path)['npx_module'])

  sample_list = get_sample(npx_data_manager=npx_data_manager, num_sample=num_sample)

  for i, (data, target) in enumerate(sample_list):
    value_data = (data*255).to(torch.uint8).numpy()
    value_target = target.to(torch.uint8).numpy()
    print(value_data, value_target)
    riscv_sample_value_bin_path = npx_define.get_riscv_sample_bin_path(i=i, is_spike=False)
    #save_sample(riscv_sample_value_bin_path, value_data, False, 4, 1, value_target)
    with open(riscv_sample_value_bin_path, "wb") as data_file:
      data_file.write(value_data)
      data_file.write(value_target)

  #print('npx_define.timesteps',npx_define.timesteps)
  spike_input = True
  for i, (data, target) in enumerate(sample_list):
    data, target = data.to(device), target.to('cpu')
    target = target.to(torch.uint8).numpy()
    riscv_testvector_bin_path = npx_define.get_riscv_testvector_bin_path(i=i)
    riscv_sample_spike_bin_path = npx_define.get_riscv_sample_bin_path(i=i, is_spike=True)
    print(riscv_testvector_bin_path)
    #if not riscv_testvector_bin_path.is_file():
    if True:
      num_steps = npx_define.timesteps
      if spike_input:
        input_data = spikegen.rate(data, num_steps=num_steps)
        with open(riscv_sample_spike_bin_path, 'wb') as spike_file:
          write_data_aligned_by_4bytes(spike_file, input_data, torch.int8)
          spike_file.write(target)
        #print(input_data.shape)
        #print(input_data)
      else :
        input_data = data.repeat(tuple([num_steps] + torch.ones(len(data.size()), dtype=int).tolist()))
      #spk_rec, _ = manual_forward_pass(npx_module, input_data, tv_path=riscv_testvector_bin_path)
      spk_rec = manual_forward_pass(npx_module, input_data, tv_path=riscv_testvector_bin_path)
      #spk_rec, _ = manual_forward_pass(npx_module, input_data)
      print(spk_rec.sum(0))
      inference_class_id = spk_rec.sum(0).argmax()
      print('class id from inference: ', int(inference_class_id))
      print('class id from dataset: ', int(target[0]))

debug_check_cpu_vs_gpu_result = False
debug_print_layer_outout = False

# for saving test vector
def manual_forward_pass(npx_module:NpxModule, data, tv_path:Path=None):
  spk_rec = []
  utils.reset(npx_module)
  if debug_check_cpu_vs_gpu_result:
    cmp_npx_module = copy.deepcopy(npx_module).to('cpu')
  if tv_path:
    tv_file = open(tv_path, 'wb')
  num_steps = npx_module.timesteps
  for step in range(num_steps):
    prev_layer_type = snntorch.Leaky
    last_tensor = data[step]
    cmp_tensor = data[step].to('cpu')

    for i, layer in enumerate(npx_module.layer_sequence):
      last_tensor = layer(last_tensor)
      if type(layer) == nn.AvgPool2d:
        last_tensor = last_tensor.to(torch.int32).to(torch.float)
      #if (type(layer) == nn.Conv2d) or (type(layer) == snntorch.Leaky):
      #  last_tensor = last_tensor.round()

      if tv_path:
        if type(layer)==snntorch.Leaky or (prev_layer_type==snntorch.Leaky and type(layer)==nn.Flatten):
          write_data_aligned_by_4bytes(tv_file, last_tensor, torch.int8)
        else:
          write_data_aligned_by_4bytes(tv_file, last_tensor, torch.int32)
      prev_layer_type = type(layer)

      if debug_check_cpu_vs_gpu_result:
        cmp_layer = cmp_npx_module.layer_sequence[i]
        cmp_tensor = cmp_layer(cmp_tensor).round()
        if (cmp_tensor==last_tensor.to('cpu')).all():
          print('###### ok #########################################################')
        else:
          print('###### not ok #####################################################')
          print(cmp_tensor.flatten()[0:10])
          print(last_tensor.flatten().to('cpu')[0:10])
          print(cmp_tensor.dtype)
          print(last_tensor.dtype)
          print(cmp_tensor==last_tensor.to('cpu'))

      if debug_print_layer_outout:
        if type(layer) == nn.Conv2d:
          print('@@@@@@@@@@@@@ layer', i, type(layer))
          if last_tensor.dim()>3:
            print('layer',i,last_tensor[0][0])
            print('layer',i,last_tensor[0][0].to(torch.int32))
          else :
            print('layer',i,last_tensor[0][0:20])
        print(last_tensor.shape)

    spk_rec.append(last_tensor)
    #if step == 0: break
  if tv_path:
    tv_file.close()
  return torch.stack(spk_rec)

use_cuda = True and torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
#device = torch.device("cuda:1" if use_cuda else "cpu")
device = torch.device("cpu")

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='NPX Framework')
  parser.add_argument('-cfg', '-c', nargs='+', help='app cfg file name')
  parser.add_argument('-cmd', nargs='+', help='command')
  parser.add_argument('-dataset', '-d', help='dataset directory')
  parser.add_argument('-output', '-o', help='output directory')
  parser.add_argument('-sample', '-s', help='number of sample', default=10)

  # check args
  args = parser.parse_args()
  assert args.cfg
  assert args.cmd
  assert args.output

  app_cfg_list = args.cfg
  cmd_list = args.cmd
  num_sample = int(args.sample)
  
  num_kfold = 5
  output_path = Path(args.output).absolute()
  assert output_path.is_dir(), output_path
  dataset_path = Path(args.dataset).absolute() if args.dataset else (output_path / 'dataset')
  assert dataset_path.is_dir(), dataset_path
  
  # cfg
  for app_cfg in app_cfg_list:
    app_cfg_path = Path(app_cfg)
    print(app_cfg_path)
    npx_define = NpxDefine(app_cfg_path=app_cfg_path, output_path=output_path)
    npx_data_manager = NpxDataManager(dataset_name=npx_define.dataset_name,
    dataset_path=dataset_path, num_kfold=num_kfold, resize=npx_define.input_resize)
    if 'testvector' in cmd_list:
      generate_testvector(npx_define=npx_define, npx_data_manager=npx_data_manager, num_sample=num_sample)
    else:
      assert 0, cmd_list
