import os
import argparse
import time
import shutil
import copy
from pathlib import *
from tqdm.auto import tqdm
from collections import namedtuple
from torchvision.transforms import Resize
import torch.nn as nn
import torch.nn.init as init

from npx_define import *
from npx_data_manager import *
from npx_module import *

def init_weights(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    if m.bias is not None:
      init.constant_(m.bias, 0.1)

class NpxTrainer():
  def __init__(self, module_class=NpxModule, gpu_id:str='-1'):
    if gpu_id=='-1' or (not torch.cuda.is_available()):
      device_option = 'cpu'
    else:
      device_option = f'cuda:{gpu_id}'
    self.device = torch.device(device_option)
        
    #self.num_steps_to_train = 32
    #self.loss_function = SF.ce_rate_loss()
    self.loss_function = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    self.log_interval = 100
    self.module_class = module_class
    self.scheduler = None
    self.grad_clip = 0.0
  
  @staticmethod
  def save_checkpoint(npx_module, optimizer, path:Path):
    check_point = {'npx_module': npx_module.state_dict(),
                   'optimizer': optimizer.state_dict() if optimizer else None,
                   }
    torch.save(check_point, path)

  @staticmethod
  def load_checkpoint(npx_module, optimizer, path:Path):
    check_point = torch.load(path, weights_only=False)
    npx_module.load_state_dict(check_point['npx_module'])
    if optimizer:
      optimizer.load_state_dict(check_point['optimizer'])

  def train(self, npx_define:NpxDefine, repeat_index:int, npx_data_manager:NpxDataManager, num_epochs:int):
    print('\n[TRAIN]', npx_define.app_name, repeat_index, num_epochs)
    npx_data_manager.setup_loader(repeat_index)
    npx_define.parameter_dir_path.mkdir(parents=True, exist_ok=True)
    npx_module = self.module_class(app_cfg_path=npx_define.app_cfg_path).to(self.device)
    #npx_module.apply(init_weights)
    # Optional optimizer / schedule controls, read from the cfg's [train]
    # section (NpxDefine.__init__ copies those keys onto itself). Every one of
    # them defaults to the previous behaviour, so cfgs that do not set them are
    # trained exactly as before.
    # Any option left unset falls through to the torch.optim.Adam default.
    # `betas` accepts either `betas=0.9,0.999` or `betas=(0.9, 0.999)` -- the cfg
    # parser runs ast.literal_eval on the value, so both yield a 2-tuple.
    optimizer_kwargs = {}
    lr = getattr(npx_define, 'lr', None)
    if lr is not None:
      optimizer_kwargs['lr'] = float(lr)
    betas = getattr(npx_define, 'betas', None)
    if betas is not None:
      assert isinstance(betas, (tuple, list)) and len(betas)==2, \
        f'[train] betas must be two values, e.g. "betas=0.9,0.999" (got {betas!r})'
      optimizer_kwargs['betas'] = tuple(float(b) for b in betas)
    weight_decay = getattr(npx_define, 'weight_decay', None)
    if weight_decay is not None:
      optimizer_kwargs['weight_decay'] = float(weight_decay)
    self.optimizer = torch.optim.Adam(npx_module.parameters(), **optimizer_kwargs)

    schedule = str(getattr(npx_define, 'lr_schedule', 'none')).lower()
    if schedule=='none':
      self.scheduler = None
    elif schedule=='cosine':
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer, T_max=num_epochs, eta_min=float(getattr(npx_define, 'lr_min', 0.0)))
    elif schedule=='step':
      self.scheduler = torch.optim.lr_scheduler.StepLR(
        self.optimizer, step_size=int(getattr(npx_define, 'lr_step', 10)),
        gamma=float(getattr(npx_define, 'lr_gamma', 0.1)))
    else:
      assert 0, schedule

    # Gradient-norm clipping. A dead SNN has exactly zero weight gradient, so a
    # loss spike that silences the network is unrecoverable; clipping keeps a
    # single bad batch from pushing it there.
    self.grad_clip = float(getattr(npx_define, 'grad_clip', 0.0))

    previous_epoch_index = -1
    previous_history_file = None
    for history_parameter_path in npx_define.parameter_dir_path.glob(npx_define.get_parameter_filename_pattern(repeat_index, False)):
      epoch_index = npx_define.get_epoch_index_from_parameter_path(history_parameter_path)
      if epoch_index > previous_epoch_index:
        previous_epoch_index = epoch_index
        previous_history_file = history_parameter_path

    start_epoch_index = previous_epoch_index + 1
    if previous_epoch_index>=0:
      self.load_checkpoint(npx_module,self.optimizer,previous_history_file)
      #npx_module.load_state_dict(torch.load(previous_history_file))
      print(f'Start from \"{previous_history_file.name}\"')
      # The scheduler is not checkpointed; fast-forward it so a resumed run
      # continues on the same LR curve.
      if self.scheduler:
        for _ in range(start_epoch_index):
          self.scheduler.step()

    # A silent SNN has exactly zero weight gradient, so the silent state is
    # absorbing. Starting straight into augmented data can drive the network
    # there before it has found any class signal, so augmentation can be delayed
    # by `augmentation_start_epoch` epochs. Default 0 = augment from the start.
    aug_start = int(getattr(npx_define, 'augmentation_start_epoch', 0))

    # Collapse guard. A silent SNN has exactly zero weight gradient, so once a
    # loss spike drives the network into the silent state it can never recover
    # on its own -- the run is dead for every remaining epoch. When enabled,
    # detect the collapse, roll back to the best checkpoint so far and halve the
    # learning rate, which is what makes the spike survivable.
    collapse_guard = str(getattr(npx_define, 'collapse_guard', False)) not in ('False', 'false', '0', 'None')
    best_acc = 0.0
    best_state = None

    for epoch_index in range(start_epoch_index, num_epochs):
      npx_data_manager.set_augmentation(epoch_index >= aug_start)
      npx_module.backup_cfg(npx_define, epoch_index)
      if self.scheduler:
        print(f'Epoch {epoch_index} lr={self.optimizer.param_groups[0]["lr"]:.6g}'
              f' aug={epoch_index >= aug_start}')
      self.train_once(npx_module=npx_module, npx_data_manager=npx_data_manager, epoch_index=epoch_index)
      if self.scheduler:
        self.scheduler.step()
      result = self.test_once(npx_module, npx_data_manager.test_loader, npx_data_manager.data_format)

      if collapse_guard:
        # Keyed on VALIDATION, never on test, so nothing from the test set can
        # influence training. Detects the collapse just as well: it is a fall to
        # near-chance, not a subtle regression.
        val_result = self.test_once(npx_module, npx_data_manager.val_loader, npx_data_manager.data_format)
        accuracy = val_result.acc / val_result.total
        if best_state is not None and accuracy < (0.5 * best_acc):
          new_lr = self.optimizer.param_groups[0]['lr'] * 0.5
          print(f'[collapse guard] epoch {epoch_index} val {accuracy:.4f} < half of best '
                f'{best_acc:.4f}; rolling back and setting lr={new_lr:.6g}')
          npx_module.load_state_dict(best_state['npx_module'])
          self.optimizer.load_state_dict(best_state['optimizer'])
          for group in self.optimizer.param_groups:
            group['lr'] = new_lr
          if self.scheduler:
            # keep the schedule going from the reduced level
            self.scheduler.base_lrs = [lr * 0.5 for lr in self.scheduler.base_lrs]
          result = self.test_once(npx_module, npx_data_manager.test_loader, npx_data_manager.data_format)
        elif accuracy > best_acc:
          best_acc = accuracy
          best_state = copy.deepcopy({'npx_module': npx_module.state_dict(),
                                      'optimizer': self.optimizer.state_dict()})

      self.save_checkpoint(npx_module, self.optimizer, npx_define.get_parameter_path(repeat_index,epoch_index, False))
      #torch.save(npx_module.state_dict(), npx_define.get_parameter_path(repeat_index,epoch_index, False))
      NpxDefine.print_test_result(result)

  def train_once(self, npx_module:NpxModule, npx_data_manager:NpxDataManager, epoch_index:int):
    npx_module.train()
  
    for batch_idx, (data, target) in enumerate(tqdm(npx_data_manager.train_loader)):
      data, target = data.to(self.device), target.to(self.device)
      if npx_module.input_size != data.shape[-2:]:
        if data.dim() == 5:
          size = (data.shape[-3],) + npx_module.input_size
        else:
          size = npx_module.input_size
        data = nn.functional.interpolate(data, size=size)
        #data = nn.functional.interpolate(data, size=size, mode='bilinear')

      spk_rec = self.forward_pass(npx_module, data, npx_data_manager.data_format)
      loss_val = self.loss_function(spk_rec, target)
      
      self.optimizer.zero_grad()
      loss_val.backward()
      #loss_val.backward(retain_graph=True)
      if getattr(self, 'grad_clip', 0.0) > 0:
        torch.nn.utils.clip_grad_norm_(npx_module.parameters(), self.grad_clip)
      self.optimizer.step()

      if (batch_idx % self.log_interval) == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch_index, batch_idx * len(data), len(npx_data_manager.train_loader.dataset),
          100. * batch_idx / len(npx_data_manager.train_loader), loss_val.item()))
  
  def test_once(self, npx_module:NpxModule, data_loader, data_format):
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
        if npx_module.input_size != data.shape[-2:]:
          if data.dim() == 5:
            size = (data.shape[-3],) + npx_module.input_size
          else:
            size = npx_module.input_size
          data = nn.functional.interpolate(data, size=size)
          #data = nn.functional.interpolate(data, size=size, mode='bilinear')
        cur = time.time()
        spk_rec = self.forward_pass(npx_module, data, data_format)

        acc += SF.accuracy_rate(spk_rec, target) * spk_rec.size(1)

        total_time += time.time() - cur
        total += spk_rec.size(1)
    return TestResult(acc, total, total_time, model_size)

  def forward_pass(self, npx_module:NpxModule, data, data_format=DataFormat.MATRIX3D):
    spk_rec = []
    utils.reset(npx_module)  # resets hidden states for all LIF neurons in net

    #num_steps = self.num_steps_to_train
    num_steps = npx_module.timesteps
    if data_format==DataFormat.MATRIX4D:
      for step in range(num_steps):
        spk_out = npx_module(data[step])
        spk_rec.append(spk_out)
    elif data_format==DataFormat.MATRIX3D:
      for step in range(num_steps):
        spk_out = npx_module(data)
        spk_rec.append(spk_out)
    else:
      assert(0)

    return torch.stack(spk_rec)

  def quantize(self, npx_define:NpxDefine, repeat_index:int):
    print('\n[QUANTIZE]', npx_define.app_name, repeat_index)
    npx_module = self.module_class(app_cfg_path=npx_define.app_cfg_path).to(self.device)
    npx_module.eval()
    for history_parameter_path in npx_define.parameter_dir_path.glob(npx_define.get_parameter_filename_pattern(repeat_index, False)):
      self.load_checkpoint(npx_module,None,history_parameter_path)
      #npx_module.load_state_dict(torch.load(history_parameter_path))
      float_parameter_text_path = npx_define.rename_path_to_parameter_text(history_parameter_path)
      if not float_parameter_text_path.is_file():
        npx_module.write_parameter(float_parameter_text_path)
      npx_module.quantize_network()
      quant_parameter_text_path = npx_define.rename_path_to_quant(float_parameter_text_path)
      if not quant_parameter_text_path.is_file():
        npx_module.write_parameter(quant_parameter_text_path)
      quant_parameter_path = npx_define.rename_path_to_quant(history_parameter_path)
      if not quant_parameter_path.is_file():
        self.save_checkpoint(npx_module, None, quant_parameter_path)
        #torch.save(npx_module.state_dict(), quant_parameter_path)

  @staticmethod
  def format_test_result(npx_define:NpxDefine, neuron_type_str, repeat_index:int, epoch_index:int, val_result:TestResult, test_result:TestResult):
    result = RecordResult(npx_define.dataset_name, neuron_type_str, neuron_type_str,
                          f'{repeat_index:01}', f'{epoch_index:03}', 
                          f'{(val_result.acc/val_result.total):.4f}',f'{(test_result.acc/test_result.total):.4f}')
    return '|'.join(result)

  def test(self, npx_define:NpxDefine, repeat_index:int, npx_data_manager:NpxDataManager):
    npx_define.report_dir_path.mkdir(parents=True, exist_ok=True)
    print('\n[TEST]', npx_define.app_name, repeat_index)

    report_path = npx_define.get_report_path(repeat_index)
    if report_path.is_file():
      npx_module = None
    else:
      result_list = []
      npx_data_manager.setup_loader(repeat_index)
      npx_module = self.module_class(app_cfg_path=npx_define.app_cfg_path).to(self.device)
      for history_parameter_path in sorted(npx_define.parameter_dir_path.glob(npx_define.get_parameter_filename_pattern(repeat_index, True)),reverse=True):
        self.load_checkpoint(npx_module,None,history_parameter_path)
        #npx_module.load_state_dict(torch.load(history_parameter_path))
        val_result = self.test_once(npx_module, npx_data_manager.val_loader, npx_data_manager.data_format)
        self.load_checkpoint(npx_module,None,history_parameter_path)
        #npx_module.load_state_dict(torch.load(history_parameter_path))
        test_result = self.test_once(npx_module, npx_data_manager.test_loader, npx_data_manager.data_format)
        epoch_index = npx_define.get_epoch_index_from_parameter_path(history_parameter_path)
        result_list.append((epoch_index,val_result, test_result))
      line_list = []
      for epoch_index, val_result, test_result in result_list:
        line_list.append(NpxTrainer.format_test_result(npx_define, npx_module.global_config('neuron_type'), repeat_index, epoch_index, val_result, test_result))
      npx_define.get_report_path(repeat_index).write_text('\n'.join(line_list))
    return npx_module

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='NPX Framework')
  parser.add_argument('-cfg', '-c', nargs='+', help='app cfg file name')
  parser.add_argument('-cmd', nargs='+', help='command')
  parser.add_argument('-dataset', '-d', help='dataset directory')
  parser.add_argument('-output', '-o', help='output directory')
  parser.add_argument('-gpu', '-g', default='-1', type=str, help='gpu id or -1 for cpu')

  # check args
  args = parser.parse_args()
  assert args.cfg
  assert args.cmd
  assert args.output
  assert args.gpu

  app_cfg_list = args.cfg
  cmd_list = args.cmd
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
    npx_define = NpxDefine(app_cfg_path=app_cfg_path, output_path=output_path)
    npx_data_manager = NpxDataManager(npx_define=npx_define, dataset_path=dataset_path, kfold=npx_define.kfold)
    if 'reset' in cmd_list:
      if npx_define.app_dir_path.is_dir():
        shutil.rmtree(npx_define.app_dir_path)
    if 'train' in cmd_list:
      for repeat_index in range(npx_define.repeat):
        npx_trainer.train(npx_define=npx_define, npx_data_manager=npx_data_manager, repeat_index=repeat_index, num_epochs=npx_define.epoch)
    if 'quantize' in cmd_list:
      for repeat_index in range(npx_define.repeat):
        npx_trainer.quantize(npx_define=npx_define, repeat_index=repeat_index)
    if 'test' in cmd_list:
      for repeat_index in range(npx_define.repeat):
        npx_trainer.test(npx_define=npx_define, npx_data_manager=npx_data_manager, repeat_index=repeat_index)
