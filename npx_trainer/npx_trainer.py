import os
import argparse
import time
import shutil
from pathlib import *
from tqdm.auto import tqdm
from collections import namedtuple

from npx_define import *
from npx_data_manager import *
from npx_module import *

class NpxTrainer():
  def __init__(self, module_class=NpxModule, gpu_id:str='-1'):
    print(gpu_id)
    if gpu_id=='-1' or (not torch.cuda.is_available()):
      device_option = 'cpu'
    else:
      device_option = f'cuda:{gpu_id}'
    self.device = torch.device(device_option)
        
    #self.num_steps_to_train = 32
    self.loss_function = SF.ce_rate_loss()
    self.log_interval = 100
    self.module_class = module_class

  def train(self, npx_define:NpxDefine, repeat_index:int, npx_data_manager:NpxDataManager, num_epochs:int):
    print('\n[TRAIN]', npx_define.app_name, npx_define.train_neuron_str, repeat_index, num_epochs)
    npx_data_manager.setup_loader(repeat_index)
    npx_define.parameter_dir_path.mkdir(parents=True, exist_ok=True)
    npx_module = self.module_class(app_cfg_path=npx_define.app_cfg_path, neuron_type_str=npx_define.train_neuron_str).to(self.device)
    #print(npx_module)
    #print(npx_module.layer_sequence)

    previous_epoch_index = -1
    previous_history_file = None
    #print(npx_define.get_parameter_filename_pattern(repeat_index, False))
    for history_parameter_path in npx_define.parameter_dir_path.glob(npx_define.get_parameter_filename_pattern(repeat_index, False)):
      #print(history_parameter_path)
      epoch_index = npx_define.get_epoch_index_from_parameter_path(history_parameter_path)
      if epoch_index > previous_epoch_index:
        previous_epoch_index = epoch_index
        previous_history_file = history_parameter_path

    start_epoch_index = previous_epoch_index + 1
    if previous_epoch_index>=0:
      npx_module.load_state_dict(torch.load(previous_history_file))
      print(f'Start from \"{previous_history_file.name}\"')

    for epoch_index in range(start_epoch_index, num_epochs):
      npx_module.backup_cfg(npx_define, epoch_index)
      self.train_once(npx_module=npx_module, npx_data_manager=npx_data_manager, epoch_index=epoch_index)
      torch.save(npx_module.state_dict(), npx_define.get_parameter_path(repeat_index,epoch_index, False))
      result = self.test_once(npx_module, npx_data_manager.test_loader)
      NpxDefine.print_test_result(result)

  def train_once(self, npx_module:NpxModule, npx_data_manager:NpxDataManager, epoch_index:int):
    npx_module.train()
    #print(npx_module.parameters())
    optimizer = torch.optim.Adam(npx_module.parameters())
  
    for batch_idx, (data, target) in enumerate(tqdm(npx_data_manager.train_loader)):
      data, target = data.to(self.device), target.to(self.device)

      spk_rec, _ = self.forward_pass(npx_module, data)
      loss_val = self.loss_function(spk_rec, target)
      
      optimizer.zero_grad()
      loss_val.backward()
      #loss_val.backward(retain_graph=True)
      optimizer.step()

      if (batch_idx % self.log_interval) == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch_index, batch_idx * len(data), len(npx_data_manager.train_loader.dataset),
          100. * batch_idx / len(npx_data_manager.train_loader), loss_val.item()))
  
  def test_once(self, npx_module:NpxModule, data_loader):
    npx_module.eval()
    total = 0
    acc = 0
    total_time = 0
    model_size = 0
    #torch.save(npx_module.state_dict(), "tmp.pth")
    #model_size = os.path.getsize("tmp.pth") / 1e6
    #os.remove("tmp.pth")
    with torch.no_grad():
      for data, target in tqdm(data_loader):
        data, target = data.to(self.device), target.to(self.device)
        
        cur = time.time()
        spk_rec, _ = self.forward_pass(npx_module, data)

        acc += SF.accuracy_rate(spk_rec, target) * spk_rec.size(1)

        total_time += time.time() - cur
        total += spk_rec.size(1)
    return TestResult(acc, total, total_time, model_size)

  def forward_pass(self, npx_module:NpxModule, data):
    mem_rec = []
    spk_rec = []
    utils.reset(npx_module)  # resets hidden states for all LIF neurons in net

    #num_steps = self.num_steps_to_train
    num_steps = npx_module.timesteps
    for step in range(num_steps):
      spk_out, mem_out = npx_module(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

  def quantize(self, npx_define:NpxDefine, repeat_index:int):
    print('\n[QUANTIZE]', npx_define.app_name, npx_define.test_neuron_str, repeat_index)
    npx_module = self.module_class(app_cfg_path=npx_define.app_cfg_path, neuron_type_str=npx_define.test_neuron_str).to(self.device)
    npx_module.eval()
    for history_parameter_path in npx_define.parameter_dir_path.glob(npx_define.get_parameter_filename_pattern(repeat_index, False)):
      npx_module.load_state_dict(torch.load(history_parameter_path))
      float_parameter_text_path = npx_define.rename_path_to_parameter_text(history_parameter_path)
      if not float_parameter_text_path.is_file():
        npx_module.write_parameter(float_parameter_text_path)
      npx_module.quantize_network()
      quant_parameter_text_path = npx_define.rename_path_to_quant(float_parameter_text_path)
      if not quant_parameter_text_path.is_file():
        npx_module.write_parameter(quant_parameter_text_path)
      quant_parameter_path = npx_define.rename_path_to_quant(history_parameter_path)
      if not quant_parameter_path.is_file():
        torch.save(npx_module.state_dict(), quant_parameter_path)

  @staticmethod
  def format_test_result(npx_define:NpxDefine, repeat_index:int, epoch_index:int, val_result:TestResult, test_result:TestResult):
    result = RecordResult(npx_define.dataset_name,npx_define.train_neuron_str,npx_define.test_neuron_str,
                          f'{repeat_index:01}', f'{epoch_index:03}', 
                          f'{(val_result.acc/val_result.total):.4f}',f'{(test_result.acc/test_result.total):.4f}')
    return '|'.join(result)

  def test(self, npx_define:NpxDefine, repeat_index:int, npx_data_manager:NpxDataManager):
    npx_define.report_dir_path.mkdir(parents=True, exist_ok=True)
    print('\n[TEST]', npx_define.app_name, npx_define.test_neuron_str, repeat_index)

    report_path = npx_define.get_report_path(repeat_index)
    if report_path.is_file():
      npx_module = None
    else:
      result_list = []
      npx_data_manager.setup_loader(repeat_index)
      npx_module = self.module_class(app_cfg_path=npx_define.app_cfg_path, neuron_type_str=npx_define.test_neuron_str).to(self.device)
      for history_parameter_path in sorted(npx_define.parameter_dir_path.glob(npx_define.get_parameter_filename_pattern(repeat_index, True)),reverse=True):
        npx_module.load_state_dict(torch.load(history_parameter_path))
        val_result = self.test_once(npx_module, npx_data_manager.val_loader)
        npx_module.load_state_dict(torch.load(history_parameter_path))
        test_result = self.test_once(npx_module, npx_data_manager.test_loader)
        epoch_index = npx_define.get_epoch_index_from_parameter_path(history_parameter_path)
        result_list.append((epoch_index,val_result, test_result))
      line_list = []
      for epoch_index, val_result, test_result in result_list:
        line_list.append(NpxTrainer.format_test_result(npx_define, repeat_index, epoch_index, val_result, test_result))
      npx_define.get_report_path(repeat_index).write_text('\n'.join(line_list))
    return npx_module

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='NPX Framework')
  parser.add_argument('-cfg', '-c', nargs='+', help='app cfg file name')
  parser.add_argument('-cmd', nargs='+', help='command')
  parser.add_argument('-epoch', '-e', help='number of epoch')
  parser.add_argument('-kfold', '-k', default=5, help='number of k-fold')
  parser.add_argument('-repeat', '-r', default=1, help='number of repeat')
  #parser.add_argument('-test_index', '-ti', help='')
  parser.add_argument('-dataset', '-d', help='dataset directory')
  parser.add_argument('-output', '-o', help='output directory')
  parser.add_argument('-cfg_dir', '-p', help='app cfg directory')
  parser.add_argument('-gpu', '-g', default='-1', type=str, help='gpu id or -1 for cpu')

  # check args
  args = parser.parse_args()
  assert args.cfg
  assert args.cmd
  assert args.epoch
  assert args.output
  assert args.gpu

  app_cfg_list = args.cfg
  cmd_list = args.cmd
  num_epochs = int(args.epoch)
  
  num_kfold = int(args.kfold)
  num_repeat = int(args.repeat)
  output_path = Path(args.output).absolute()
  if not output_path.is_dir():
    output_path.mkdir(parents=True)
  dataset_path = Path(args.dataset).absolute() if args.dataset else (output_path / 'dataset')

  # common env
  torch.manual_seed(1)
  npx_trainer = NpxTrainer(gpu_id=args.gpu)

  # cfg
  for app_cfg in app_cfg_list:
    app_cfg_path = Path(app_cfg)
    #print(app_cfg_path)
    npx_define = NpxDefine(app_cfg_path=app_cfg_path, output_path=output_path)
    npx_data_manager = NpxDataManager(dataset_name=npx_define.dataset_name, dataset_path=dataset_path, num_kfold=num_kfold)
    if 'reset' in cmd_list:
      if npx_define.app_dir_path.is_dir():
        shutil.rmtree(npx_define.app_dir_path)
    if 'train' in cmd_list:
      for repeat_index in range(num_repeat):
        npx_trainer.train(npx_define=npx_define, npx_data_manager=npx_data_manager, repeat_index=repeat_index, num_epochs=num_epochs)
    if 'quantize' in cmd_list:
      for repeat_index in range(num_repeat):
        npx_trainer.quantize(npx_define=npx_define, repeat_index=repeat_index)
    if 'test' in cmd_list:
      for repeat_index in range(num_repeat):
        npx_trainer.test(npx_define=npx_define, npx_data_manager=npx_data_manager, repeat_index=repeat_index)
