import os
import argparse
import time
import shutil
import random

import numpy as np
from torch.utils.data.dataloader import DataLoader
from pathlib import *
from tqdm.auto import tqdm
from collections import namedtuple

from npx_define import *
from npx_data_manager import *
from npx_module import *

#from npx_data_loader import *

#from npx_framework import NpxFramework

class NpxConverter():
  def __init__(self, use_cuda:bool=None):
    self.use_cuda = (use_cuda!=None) and torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")
    self.num_steps_to_train = 32

  def binary(self, npx_define:NpxDefine, repeat_index:int):
    print('\n[BINARY]', npx_define.app_name, npx_define.test_neuron_str, repeat_index)
    npx_module = NpxModule(net_cfg_path=npx_define.net_cfg_path, 
                           neuron_type_str=npx_define.train_neuron_str).to(self.device)

    for history_cfg_path in sorted(npx_define.neuron_dir_path.glob(npx_define.get_cfg_filename_pattern(repeat_index, True)),reverse=True):
      print('\n[load model]', history_cfg_path)
      npx_module.load_state_dict(torch.load(history_cfg_path))
      bin_path = self.rename_path_to_cfg_bin(history_cfg_path)
      if not bin_path.is_file():
        self.write_parameter_to_binaryfile(npx_module=npx_module, bin_path=bin_path)

  def test_vector(self, npx_define:NpxDefine, repeat_index:int,
            npx_data_manager:NpxDataManager, spike_input=True):
    print('\n[TEST VECTOR]', npx_define.app_name, npx_define.test_neuron_str, repeat_index)
    npx_data_manager.setup_loader(repeat_index)
    data_loader = DataLoader(npx_data_manager.dataset_test, batch_size=1, shuffle=False, drop_last=False)
    npx_module = NpxModule(net_cfg_path=npx_define.net_cfg_path, 
                           neuron_type_str=npx_define.train_neuron_str).to(self.device)
    npx_module.eval()
    for history_cfg_path in sorted(npx_define.neuron_dir_path.glob(npx_define.get_cfg_filename_pattern(repeat_index, True)),reverse=True):
      npx_module.load_state_dict(torch.load(history_cfg_path))
      tv_path = self.rename_path_to_test_vector(history_cfg_path)
      if not tv_path.is_file():
      #if True:
        data, target = next(iter(data_loader))
        data = data.to(self.device)
        target = target.to(self.device)
        num_steps = self.num_steps_to_train
        if spike_input:
          input_data = spikegen.rate(data, num_steps=num_steps)
        else :
          input_data = data.repeat(tuple([num_steps] + torch.ones(len(data.size()), dtype=int).tolist()))
        spk_rec, _ = self.manual_forward_pass(npx_module, input_data, tv_path=tv_path)
        print(spk_rec.sum(0))
        inference_class_id = spk_rec.sum(0).argmax()
        print('class id from inference: ', int(inference_class_id))
        print('class id from dataset: ', int(target))
      break
    
  def write_parameter_to_binaryfile(self, npx_module:NpxModule, bin_path:Path):
    print('save bin file:', bin_path)
    # print(npx_module.state_dict())
    with open(bin_path, "wb") as bin_file:
      for i, (layer, neuron) in enumerate(npx_module.layer_sequence):
        weights = layer.weight.data.flatten()
        threshold = neuron.threshold
        self.write_data_aligned_by_4bytes(bin_file, weights, torch.int8)
        self.write_data_aligned_by_4bytes(bin_file, threshold, torch.int32)

  def write_data_aligned_by_4bytes(self, file_io, data, data_type):
    data = data.to(data_type).numpy().reshape(-1)
    lenth = data.shape[0]
    fill_len = 0
    if (data_type == torch.int8) | (data_type == torch.uint8) :
      if (lenth%4) > 0 :
        fill_len = 4 - (lenth%4)
    elif (data_type == torch.int16) :
      if (lenth%2) > 0 :
        fill_len = 2 - (lenth%2)
    elif (data_type == torch.int32) :
      fill_len = 0
    else :
      print(f'unsupported type {data_type} in write_data_aligned_by_4bytes')
      return
    if fill_len > 0:
      fill_data = np.zeros(fill_len, dtype=data.dtype)
      data = np.append(data, fill_data)
    file_io.write(data)

  # for saving test vector
  def manual_forward_pass(self, npx_module:NpxModule, data, tv_path:Path=None):
    mem_rec = []
    spk_rec = []
    utils.reset(npx_module)
    if tv_path:
      tv_file = open(tv_path,'wb')
    num_steps = self.num_steps_to_train
    for step in range(num_steps):
    #for step in range(10):
      last_tensor = data[step]
      # print(f'### step:{step}')
      for i, (layer, neuron) in enumerate(npx_module.layer_sequence):
        if type(layer)==nn.Linear:
          last_tensor = torch.flatten(last_tensor, 1)
        current = layer(last_tensor)
        if tv_path:
          self.write_data_aligned_by_4bytes(tv_file, last_tensor,
          torch.int8)
          self.write_data_aligned_by_4bytes(tv_file, current, torch.int32)
        last_tensor = neuron(current)
        # print('current', current)
        # print('mem', neuron.mem)
        # print('spike out', last_tensor)
      spk_out, mem_out = last_tensor
      if tv_path:
        self.write_data_aligned_by_4bytes(tv_file, spk_out, torch.int8)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)
    if tv_path:
      tv_file.close()
    return torch.stack(spk_rec), torch.stack(mem_rec)

  def rename_path_to_cfg_bin(self, path:Path):
    assert path.suffix=='.pt', path
    return path.parent / f'{path.stem}.bin'
  
  def rename_path_to_test_vector(self, path:Path):
    assert path.suffix=='.pt', path
    return path.parent / f'{path.stem}.tv'   

  '''
  def save_image_as_binary(self, npx_define:NpxDefine, dataset_path:Path, data_type:str, num=20):
    npx_data_loader = NpxDataLoader(npx_define.dataset_name, dataset_path, 
                                         data_type=data_type, batch_size=1)
    cnt = 0
    for idx, (data, target) in enumerate(tqdm(npx_data_loader.test_loader)):
      data = data.numpy().astype(data_type)
      target = target.numpy().astype('uint8')
      if target == (cnt%10) :
        img_path = self.get_image_filename(npx_data_loader.download_path, data_type, int(target), cnt)
        print(img_path)
        print(data)
        # print(target)
        with open(img_path, "wb") as img_file:
          img_file.write(data)
          img_file.write(target)

        # with open(img_path, "rb") as img_file:
        #   rdata = img_file.read()
        rdata = np.fromfile(img_path, dtype=np.uint8)
        print(rdata)
        cnt += 1
      if cnt == 1: break

  def get_image_filename(self, path:Path, data_type:str, target:int, index:int):
    rawimage_path = path / f'image_{data_type}_c{target}_{index:05}.bin'
    return rawimage_path

  # dodododododo
  def save_spike_to_binaryfile(self, npx_data_manager:NpxDataManager):
    data_path = npx_data_manager.download_path / f'data_int8.bin'
    target_path = npx_data_manager.download_path / f'target_int8.bin'
    data_loader = DataLoader(npx_data_manager.dataset_test, batch_size=1, 
                                            shuffle=False, drop_last=False)
    with open(data_path, "wb") as data_file, open(target_path, "wb") as target_file:
      for data, target in tqdm(data_loader):
        #data, target = data.to(self.device), target.to(self.device)
        data = data.to(torch.float32).numpy()
        target = target.to(torch.uint8).numpy()
        data_file.write(data)
        target_file.write(target)
        # print(data)
        # print(data.dtype)
        # print(target)
        # print(target.dtype)
        break
  '''

  def inference_one_image(self, npx_define:NpxDefine, repeat_index:int,
            npx_data_manager:NpxDataManager, spike_input=True):
    print('inference_one_image')
    npx_data_manager.setup_loader(repeat_index)
    data_loader = DataLoader(npx_data_manager.dataset_test, batch_size=1,
                             shuffle=True, drop_last=False)
    npx_module = NpxModule(net_cfg_path=npx_define.net_cfg_path, 
                           neuron_type_str=npx_define.train_neuron_str).to(self.device)
    for history_cfg_path in sorted(npx_define.neuron_dir_path.glob(npx_define.get_cfg_filename_pattern(repeat_index, True)),reverse=True):
      print(history_cfg_path)
      npx_module.load_state_dict(torch.load(history_cfg_path))
      npx_module.eval()
      data, target = next(iter(data_loader))
      data = data.to(self.device)
      target = target.to(self.device)
      num_steps = self.num_steps_to_train
      if spike_input:
        input_data = spikegen.rate(data, num_steps=num_steps)
      else :
        input_data = data.repeat(tuple([num_steps] + torch.ones(len(data.size()), dtype=int).tolist()))
      spk_rec, _ = self.manual_forward_pass(npx_module, input_data)
      # spk_rec, _ = self.forward_pass(npx_module, input_data)
      inference_class_id = spk_rec.sum(0).argmax()
      print('class id from inference: ', int(inference_class_id))
      print('class id from dataset: ', int(target))
      print('\n')
      break

  # from class NpxFramework()
  def forward_pass(self, npx_module:NpxModule, data):
    mem_rec = []
    spk_rec = []
    utils.reset(npx_module)  # resets hidden states for all LIF neurons in net

    num_steps = self.num_steps_to_train
    for step in range(num_steps):
      spk_out, mem_out = npx_module(data[step])
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

  # direct input vs rate coding input
  def sim(self, npx_define:NpxDefine, repeat_index:int, npx_data_manager:NpxDataManager):
    print('\n[TEST direct input vs rate-coding input]', npx_define.app_name, npx_define.test_neuron_str, repeat_index)
    npx_data_manager.setup_loader(repeat_index)
    for history_cfg_path in sorted(npx_define.neuron_dir_path.glob(npx_define.get_cfg_filename_pattern(repeat_index, True)),reverse=True):
      if history_cfg_path.is_file():
        print(history_cfg_path)
        npx_module = self.load_model_from_path(npx_define=npx_define, parameter_path=history_cfg_path)
        test_result = self.test_once(npx_module, npx_data_manager.test_loader, spike_input=False)
        print(f'[Direct input] Accuracy: {(test_result.acc/test_result.total):.4f} / Time: {(test_result.total_time):.4f} sec')
        npx_module = self.load_model_from_path(npx_define=npx_define, parameter_path=history_cfg_path)
        test_result = self.test_once(npx_module, npx_data_manager.test_loader, spike_input=True)
        print(f'[Rate-coding input] Accuracy: {(test_result.acc/test_result.total):.4f} / Time: {(test_result.total_time):.4f} sec')
        # break

  def test_once(self, npx_module:NpxModule, data_loader, spike_input=False):
    npx_module.eval()
    total = 0
    acc = 0
    total_time = 0
    torch.save(npx_module.state_dict(), "tmp.pth")
    model_size = os.path.getsize("tmp.pth") / 1e6
    os.remove("tmp.pth")
    with torch.no_grad():
      for data, target in tqdm(data_loader):
        data, target = data.to(self.device), target.to(self.device)
        num_steps = self.num_steps_to_train
        if spike_input:
          input_data = spikegen.rate(data, num_steps=num_steps)
        else :
          input_data = data.repeat(tuple([num_steps] + torch.ones(len(data.size()), dtype=int).tolist()))
        cur = time.time()
        spk_rec, _ = self.forward_pass(npx_module, input_data)
        # spk_rec, _ = self.manual_forward_pass(npx_module, input_data)
        acc += SF.accuracy_rate(spk_rec, target) * spk_rec.size(1)
        total_time += time.time() - cur
        total += spk_rec.size(1)
        # break
    return TestResult(acc, total, total_time, model_size)

  def load_model_from_path(self, npx_define:NpxDefine, parameter_path:Path):
    print('\n[load model]', parameter_path)
    assert parameter_path.is_file()
    npx_module = NpxModule(net_cfg_path=npx_define.net_cfg_path, 
                           neuron_type_str=npx_define.train_neuron_str).to(self.device)
    npx_module.load_state_dict(torch.load(parameter_path))
    return npx_module