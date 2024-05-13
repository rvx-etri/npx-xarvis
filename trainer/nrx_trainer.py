import os
import argparse
import time
import shutil
from pathlib import *
from tqdm.auto import tqdm
from collections import namedtuple

from nrx_define import *
from nrx_data_manager import *
from nrx_module import *

class NrxTrainer():
  def __init__(self, use_cuda:bool=None):
    self.use_cuda = (use_cuda!=None) and torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")
    self.num_steps_to_train = 32
    self.loss_function = SF.ce_rate_loss()
    self.log_interval = 100

  def train(self, nrx_define:NrxDefine, repeat_index:int, nrx_data_manager:NrxDataManager, num_epochs:int):
    print('\n[TRAIN]', nrx_define.app_name, nrx_define.train_neuron_str, repeat_index, num_epochs)
    nrx_data_manager.setup_loader(repeat_index)
    nrx_define.neuron_dir_path.mkdir(parents=True, exist_ok=True)
    nrx_module = NrxModule(app_name=nrx_define.app_name, neuron_type_str=nrx_define.train_neuron_str).to(self.device)

    previous_epoch_index = -1
    previous_history_file = None
    for history_cfg_path in nrx_define.neuron_dir_path.glob(nrx_define.get_cfg_filename_pattern(repeat_index, False)):
      epoch_index = nrx_define.get_epoch_index_from_cfg_path(history_cfg_path)
      if epoch_index > previous_epoch_index:
        previous_epoch_index = epoch_index
        previous_history_file = history_cfg_path

    start_epoch_index = previous_epoch_index + 1
    if previous_epoch_index>=0:
      nrx_module.load_state_dict(torch.load(previous_history_file))
      print(f'Start from \"{previous_history_file.name}\"')

    for epoch_index in range(start_epoch_index, num_epochs):
      self.train_once(nrx_module=nrx_module, nrx_data_manager=nrx_data_manager, epoch_index=epoch_index)
      torch.save(nrx_module.state_dict(), nrx_define.get_cfg_path(repeat_index,epoch_index, False))
      result = self.test_once(nrx_module, nrx_data_manager.test_loader)
      NrxDefine.print_test_result(result)

  def train_once(self, nrx_module:NrxModule, nrx_data_manager:NrxDataManager, epoch_index:int):
    nrx_module.train()
    optimizer = torch.optim.Adam(nrx_module.parameters())
  
    for batch_idx, (data, target) in enumerate(tqdm(nrx_data_manager.train_loader)):
      data, target = data.to(self.device), target.to(self.device)

      spk_rec, _ = self.forward_pass(nrx_module, data)
      loss_val = self.loss_function(spk_rec, target)
      
      optimizer.zero_grad()
      loss_val.backward()
      optimizer.step()

      if (batch_idx % self.log_interval) == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch_index, batch_idx * len(data), len(nrx_data_manager.train_loader.dataset),
          100. * batch_idx / len(nrx_data_manager.train_loader), loss_val.item()))
  
  def test_once(self, nrx_module:NrxModule, data_loader):
    nrx_module.eval()
    total = 0
    acc = 0
    total_time = 0
    torch.save(nrx_module.state_dict(), "tmp.pth")
    model_size = os.path.getsize("tmp.pth") / 1e6
    os.remove("tmp.pth")
    with torch.no_grad():
      for data, target in tqdm(data_loader):
        data, target = data.to(self.device), target.to(self.device)
        
        cur = time.time()
        spk_rec, _ = self.forward_pass(nrx_module, data)

        acc += SF.accuracy_rate(spk_rec, target) * spk_rec.size(1)

        total_time += time.time() - cur
        total += spk_rec.size(1)
    return TestResult(acc, total, total_time, model_size)

  def forward_pass(self, nrx_module:NrxModule, data):
    mem_rec = []
    spk_rec = []
    utils.reset(nrx_module)  # resets hidden states for all LIF neurons in net

    num_steps = self.num_steps_to_train
    for step in range(num_steps):
      spk_out, mem_out = nrx_module(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

  def quantize(self, nrx_define:NrxDefine, repeat_index:int):
    nrx_module = NrxModule(app_name=nrx_define.app_name, neuron_type_str=nrx_define.test_neuron_str).to(self.device)
    nrx_module.eval()
    for history_cfg_path in nrx_define.neuron_dir_path.glob(nrx_define.get_cfg_filename_pattern(repeat_index, False)):
      nrx_module.load_state_dict(torch.load(history_cfg_path))
      history_cfg_text_path = nrx_define.rename_path_to_cfg_text(history_cfg_path)
      nrx_module.write_cfg(history_cfg_text_path)
      nrx_module.quantize_network()
      nrx_module.write_cfg(nrx_define.rename_path_to_quant(history_cfg_text_path))
      torch.save(nrx_module.state_dict(), nrx_define.rename_path_to_quant(history_cfg_path))

  @staticmethod
  def format_test_result(nrx_define:NrxDefine, repeat_index:int, epoch_index:int, val_result:TestResult, test_result:TestResult):
    result = RecordResult(nrx_define.dataset_name,nrx_define.train_neuron_str,nrx_define.test_neuron_str,
                          f'{repeat_index:01}', f'{epoch_index:03}', 
                          f'{(val_result.acc/val_result.total):.4f}',f'{(test_result.acc/test_result.total):.4f}')
    return '|'.join(result)
  
  def test(self, nrx_define:NrxDefine, repeat_index:int, nrx_data_manager:NrxDataManager):
    print('\n[TEST]', nrx_define.app_name, nrx_define.test_neuron_str, repeat_index)
    self.quantize(nrx_define, repeat_index)

    report_path = nrx_define.get_report_path(repeat_index)
    if not report_path.is_file():
      result_list = []
      nrx_data_manager.setup_loader(repeat_index)
      nrx_module = NrxModule(app_name=app_name, neuron_type_str=nrx_define.test_neuron_str).to(self.device)
      for history_cfg_path in sorted(nrx_define.neuron_dir_path.glob(nrx_define.get_cfg_filename_pattern(repeat_index, True)),reverse=True):
        nrx_module.load_state_dict(torch.load(history_cfg_path))
        val_result = self.test_once(nrx_module, nrx_data_manager.val_loader)
        nrx_module.load_state_dict(torch.load(history_cfg_path))
        test_result = self.test_once(nrx_module, nrx_data_manager.test_loader)
        epoch_index = int(history_cfg_path.name.split('_')[-2])
        result_list.append((epoch_index,val_result, test_result))
      line_list = []
      for epoch_index, val_result, test_result in result_list:
        line_list.append(NrxTrainer.format_test_result(nrx_define, repeat_index, epoch_index, val_result, test_result))
      nrx_define.get_report_path(repeat_index).write_text('\n'.join(line_list))

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='NRX Framework')
  parser.add_argument('-app', '-a', nargs='+', help='app name')
  parser.add_argument('-cmd', nargs='+', help='command')
  parser.add_argument('-epoch', '-e', help='number of epoch')
  parser.add_argument('-neuron', '-n', nargs='+', help='types of neuron')
  parser.add_argument('-kfold', '-k', help='number of k-fold')
  parser.add_argument('-repeat', '-r', help='number of repeat')
  #parser.add_argument('-test_index', '-ti', help='')
  parser.add_argument('-dataset', '-d', help='dataset directory')
  parser.add_argument('-output', '-o', help='output directory')

  # check args
  args = parser.parse_args()
  assert args.app
  assert args.cmd
  assert args.epoch
  assert args.neuron
  assert args.output

  app_name_list = args.app
  cmd_list = args.cmd
  num_epochs = int(args.epoch)
  neuron_list = []
  for neuron_set in args.neuron:
    if '-' in neuron_set:
      train_neuron_str, test_neuron_str = neuron_set.split('-')
    else:
      train_neuron_str = neuron_set
      test_neuron_str = neuron_set
    neuron_list.append((train_neuron_str,test_neuron_str))
  num_kfold = int(args.kfold) if args.kfold else 5
  num_repeat = int(args.repeat) if args.repeat else 1
  output_path = Path(args.output).absolute()
  if not output_path.is_dir():
    output_path.relative_to(Path('.').absolute())
    output_path.mkdir(parents=True)
  dataset_path = Path(args.dataset).absolute() if args.dataset else (output_path / 'dataset')

  # common env
  torch.manual_seed(1)
  nrx_trainer = NrxTrainer()

  # cmd
  for app_name in app_name_list:
    for train_neuron_str, test_neuron_str in neuron_list:    
      nrx_define = NrxDefine(app_name=app_name, train_neuron_str=train_neuron_str, test_neuron_str=test_neuron_str, output_path=output_path)
      nrx_data_manager = NrxDataManager(dataset_name=nrx_define.dataset_name, dataset_path=dataset_path, num_kfold=num_kfold)
      if 'reset' in cmd_list:
        if nrx_define.neuron_dir_path.is_dir():
          shutil.rmtree(nrx_define.neuron_dir_path)
      if 'train' in cmd_list:
        for repeat_index in range(num_repeat):
          nrx_trainer.train(nrx_define=nrx_define, nrx_data_manager=nrx_data_manager, repeat_index=repeat_index, num_epochs=num_epochs)
      if 'quantize' in cmd_list:
        for repeat_index in range(num_repeat):
          nrx_trainer.quantize(nrx_define=nrx_define, repeat_index=repeat_index)
      if 'test' in cmd_list:
        for repeat_index in range(num_repeat):
          nrx_trainer.test(nrx_define=nrx_define, nrx_data_manager=nrx_data_manager, repeat_index=repeat_index)
