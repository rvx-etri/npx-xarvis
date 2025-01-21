from pathlib import *
from collections import namedtuple
from enum import Enum

from npx_cfg_parser import *

TestResult = namedtuple('TestResult', ['acc', 'total', 'total_time', 'model_size'])
RecordResult = namedtuple('RecordResult', ['dataset_name', 'train_neuron_str', 'test_neuron_str', 'repeat_index_str','epoch_index_str', 'val_accuracy_str', 'test_accuracy_str'])
#CfgFilename = namedtuple('CfgFilename', ['prefix', 'repeat_index', 'epoch_index', 'value_type'])

class DataFormat(Enum):
  MATRIX3D = 1
  MATRIX4D = 2
  DVS = 3
  
class NpxDefine:
  def __init__(self, app_cfg_path:Path, output_path:Path):
    self.app_cfg_path = app_cfg_path
    self.output_path = output_path

    self.cfg_parser = NpxCfgParser()
    self.cfg_parser.parse_file(self.app_cfg_path)
    #cfg_parser.write_file(self.app_cfg_path)
    
    preprocess_default_info_list = (
        ('dataset', 'mnist_dataset'),('timesteps', 32)
        )
    for var_name, default_value in preprocess_default_info_list:
      value = self.cfg_parser.preprocess_info.setdefault(var_name, default_value)
      setattr(self, var_name, value)
      self.__dict__[var_name] = value
    
    train_default_info_list = (
        ('neuron_type', ''),
        ('train_neuron_str', ''),('test_neuron_str', ''),
        ('input_channels', 3),('input_size', (14,14)),
        ('output_classes', 10),
        ('epoch',10),('kfold',5),('repeat',1)
        )
    for var_name, default_value in train_default_info_list:
      value = self.cfg_parser.train_info.setdefault(var_name, default_value)
      setattr(self, var_name, value)
      self.__dict__[var_name] = value
    
    if self.neuron_type:
      self.train_neuron_str = self.neuron_type
      self.test_neuron_str = self.neuron_type
    else:
      assert self.train_neuron_str
      if not self.test_neuron_str:
        self.test_neuron_str = self.train_neuron_str

  @staticmethod
  def print_test_result(result:TestResult):
    print(f"Acc: {result.acc / result.total:.4f}, Inference Time: {result.total_time / 10000}, "
              f"Model Size: {result.model_size:.2f}MB")

  @property
  def app_name(self):
    return self.app_cfg_path.stem
  
  @property
  def app_dir_path(self):
    return self.output_path / self.app_name

  @property
  def parameter_dir_path(self):
    return self.app_dir_path / 'parameter'
  
  @property
  def report_dir_path(self):
    return self.app_dir_path / 'report'
  
  @property
  def riscv_dir_path(self):
    return self.app_dir_path / 'riscv'

  @property
  def dataset_name(self):
    return self.dataset

  def get_parameter_filename_prefix(self, repeat_index:int):
    return f'{self.app_name}_r{repeat_index}'
  
  @staticmethod
  def parameter_suffix():
    return '.pt'
    
  def get_parameter_raw_cfg_path(self):
    return self.parameter_dir_path / self.app_cfg_path.name
  
  def get_parameter_epoch_cfg_path(self, epoch_index:int):
    return self.parameter_dir_path / f'{self.app_cfg_path.stem}_e{epoch_index:03d}{self.app_cfg_path.suffix}'
  
  def get_parameter_path(self, repeat_index:int, epoch_index:int, is_quantized:bool):
    filename = self.get_parameter_filename_prefix(repeat_index)
    filename += f'_e{epoch_index:03d}'
    filename += '_parameter'
    filename += '_quant' if is_quantized else '_float'
    filename += NpxDefine.parameter_suffix()
    return self.parameter_dir_path / filename
  
  def get_parameter_filename_pattern(self, repeat_index:int, is_quantized:bool):
    pattern = self.get_parameter_filename_prefix(repeat_index)
    pattern += f'_*'
    pattern += '_quant' if is_quantized else '_float'
    pattern += NpxDefine.parameter_suffix()
    return pattern
  
  @staticmethod
  def rename_path_to_parameter_text(path:Path):
    assert path.suffix==NpxDefine.parameter_suffix(), path
    return path.parent / f'{path.stem}.txt'

  @staticmethod
  def rename_path_to_quant(path:Path):
    assert 'float' in path.stem, path 
    return path.parent / path.name.replace('float', 'quant')

  def get_parameter_text_path(self, repeat_index:int, epoch_index:int, is_quantized:bool):
    return self.rename_path_to_parameter_text(self.get_parameter_path(repeat_index,epoch_index,is_quantized))
  
  @staticmethod
  def get_epoch_index_from_parameter_path(path:Path):
    return int(path.stem.split('_')[-3][1:])

  def get_test_prefix(self, repeat_index:int):
    prefix = self.get_parameter_filename_prefix(repeat_index)
    #prefix += f'_{self.test_neuron_str}'
    return prefix

  def get_report_path(self, repeat_index:int):
    return self.report_dir_path  / f'{self.get_test_prefix(repeat_index)}_accuracy.txt'
  
  def get_report_filename_pattern(self):
    pattern = self.app_name
    pattern += '_*_accuracy.txt'
    return pattern
  
  def get_analysis_path(self, repeat_index:int):
    return self.report_dir_path  / f'{self.get_test_prefix(repeat_index)}_analysis.txt'
  
  @staticmethod
  def get_repeat_index_from_report_path(path:Path):
    return int(path.stem.split('_')[-2][1:])
  
  def get_riscv_network_path(self):
    return self.riscv_dir_path / f'{self.app_cfg_path.stem}_network{self.app_cfg_path.suffix}'
  
  def get_riscv_parameter_path(self, is_quantized:bool):
    filename = self.app_name
    filename += '_parameter'
    filename += '_quant' if is_quantized else '_float'
    filename += NpxDefine.parameter_suffix()
    return self.riscv_dir_path / filename

  def get_riscv_info_path(self, is_quantized:bool):
    filename = self.app_name
    filename += '_info'
    filename += '_quant' if is_quantized else '_float'
    filename += NpxDefine.parameter_suffix()
    return self.riscv_dir_path / filename
  
  def get_riscv_info_text_path(self, is_quantized:bool):
    filename = self.app_name
    filename += '_info'
    filename += '_quant' if is_quantized else '_float'
    filename += '.txt'
    return self.riscv_dir_path / filename
  
  def get_riscv_parameter_text_path(self, is_quantized:bool):
    filename = self.app_name
    filename += '_parameter'
    filename += '_quant' if is_quantized else '_float'
    filename += '.txt'
    return self.riscv_dir_path / filename

  def get_riscv_parameter_bin_path(self, is_quantized:bool):
    filename = self.app_name
    filename += '_parameter'
    filename += '_quant' if is_quantized else '_float'
    filename += '.bin'
    return self.riscv_dir_path / filename

  def get_riscv_sample_bin_path(self, i:int, data_format:DataFormat):
    filename = self.app_name
    filename += '_sample'
    if data_format == DataFormat.MATRIX3D:
      filename += '_matrix3d'
    elif data_format == DataFormat.MATRIX4D:
      filename += '_matrix4d'
    elif data_format == DataFormat.DVS:
      filename += '_dvs'
    filename += f'_{i:03}'
    filename += '.bin'
    return self.riscv_dir_path / filename

  def get_riscv_testvector_bin_path(self, i:int):
    filename = self.app_name
    filename += '_testvector'
    filename += f'_{i:03}'
    filename += '.bin'
    return self.riscv_dir_path / filename

  def get_riscv_testvector_text_path(self, i:int):
    filename = self.app_name
    filename += '_testvector'
    filename += f'_{i:03}'
    filename += '.text'
    return self.riscv_dir_path / filename