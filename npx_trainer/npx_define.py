from pathlib import *
from collections import namedtuple
from npx_text_parser import *

TestResult = namedtuple('TestResult', ['acc', 'total', 'total_time', 'model_size'])
RecordResult = namedtuple('RecordResult', ['dataset_name', 'train_neuron_str', 'test_neuron_str', 'repeat_index_str','epoch_index_str', 'val_accuracy_str', 'test_accuracy_str'])
#CfgFilename = namedtuple('CfgFilename', ['prefix', 'repeat_index', 'epoch_index', 'value_type'])

class NpxDefine:
  def __init__(self, app_cfg_path:Path, output_path:Path):
    self.app_cfg_path = app_cfg_path
    self.app_name = self.app_cfg_path.stem
    self.output_path = output_path

    net_parser = NpxTextParser(self.app_cfg_path)
    net_parser.parsing()
    net_parser.save()
    # print(net_parser.section_list)
    net_option = net_parser.section_list[0]
    self.dataset = net_parser.find_option_value(net_option, 'dataset', 'mnist')

    self.train_neuron_str = net_parser.find_option_value(net_option, 'neuron_type', 'q8ssf')
    self.test_neuron_str = self.train_neuron_str

  @staticmethod
  def print_test_result(result:TestResult):
    print(f"Acc: {result.acc / result.total:.4f}, Inference Time: {result.total_time / 10000}, "
              f"Model Size: {result.model_size:.2f}MB")

  @property
  def app_dir_path(self):
    return self.output_path / self.app_name

  @property
  def neuron_dir_path(self):
    return self.app_dir_path / self.train_neuron_str
  
  @property
  def report_dir_path(self):
    return self.app_dir_path

  @property
  def dataset_name(self):
    return self.dataset

  @property
  def app_version(self):
    return f'{self.app_name}_{self.train_neuron_str}'

  def get_parameter_filename_prefix(self, repeat_index:int):
    return f'{self.app_version}_{repeat_index}'
  
  def get_parameter_path(self, repeat_index:int, epoch_index:int, is_quantized:bool):
    filename = self.get_parameter_filename_prefix(repeat_index)
    filename += f'_{epoch_index:03d}'
    filename += '_quant' if is_quantized else '_float'
    filename += '.pt'
    return self.neuron_dir_path / filename
  
  def get_parameter_filename_pattern(self, repeat_index:int, is_quantized:bool):
    pattern = self.get_parameter_filename_prefix(repeat_index)
    pattern += f'_*'
    pattern += '_quant' if is_quantized else '_float'
    pattern += '.pt'
    return pattern
  
  @staticmethod
  def rename_path_to_parameter_text(path:Path):
    assert path.suffix=='.pt', path
    return path.parent / f'{path.stem}.txt'

  @staticmethod
  def rename_path_to_quant(path:Path):
    assert 'float' in path.stem, path 
    return path.parent / path.name.replace('float', 'quant')

  def get_parameter_text_path(self, repeat_index:int, epoch_index:int, is_quantized:bool):
    return self.rename_path_to_parameter_text(self.get_parameter_path(repeat_index,epoch_index,is_quantized))
  
  @staticmethod
  def get_epoch_index_from_parameter_path(path:Path):
    return int(path.stem.split('_')[-2])

  def get_test_prefix(self, repeat_index:int):
    prefix = self.get_parameter_filename_prefix(repeat_index)
    prefix += f'_{self.test_neuron_str}'
    return prefix

  def get_report_path(self, repeat_index:int):
    return self.report_dir_path  / f'{self.get_test_prefix(repeat_index)}_accuracy.txt'
