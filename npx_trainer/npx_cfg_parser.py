from pathlib import *
import math
import copy
import os
import re
import ast

class NpxCfgSection(dict):
  def __init__(self, name:str=None):
    self.name = name
    
  @staticmethod
  def convert_value(text):
    try:
      return ast.literal_eval(text)
    except (ValueError, SyntaxError):
      return text
  
  def add_option(self, line:str):
    key, value = line.strip().split('=')
    self[key] = NpxCfgSection.convert_value(value)
  
  def __str__(self) -> str:
    line_list = []
    line_list.append(f'[{self.name}]')
    for key, value in self.items():
      line_list.append(f'{key}={value}')
    return '\n'.join(line_list)
  
class NpxCfgParser():
  def __init__(self, path:Path=None):
    super().__init__()
    self.find_section_name = re.compile('\[([^]]+)\]').findall
    self.train_info = None
    self.preprocess_info = None
    self.layer_info_list = []
    if path:
      self.parse_file(path)
  
  def parse_file(self, path:Path):
    assert path.is_file(), path
    line_list = path.read_text().split('\n')

    current_section = None
    for line in line_list:
      if line=='':
        continue
      if '[' in line and ']' in line:
        section_name = self.find_section_name(line)[0]
        current_section = NpxCfgSection(section_name)
        if section_name=='train':
          self.train_info = current_section
        elif section_name=='preprocess':
          self.preprocess_info = current_section
        else:
          self.layer_info_list.append(current_section)
      elif line.startswith('\0') or line.startswith('#') or line.startswith(';') or line.startswith('\n'):
          pass
      else:
        current_section.add_option(line)
  
  def __str__(self) -> str:
    line_list = []
    if self.preprocess_info:
      line_list.append(str(self.preprocess_info))
    if self.train_info:
      line_list.append(str(self.train_info))
    for section in self.layer_info_list:
      line_list.append(str(section))
    return '\n'.join(line_list)
  
  def __repr__(self) -> str:
    return str(self)+'\n' 
  
  def write_file(self, path:Path):
    path.write_text(str(self))
    
  def add_train_info(self):
    assert self.train_info
    self.train_info = NpxCfgSection('train')
    return self.train_info
  
  def add_preprocess_info(self):
    assert self.preprocess_info
    self.preprocess_info = NpxCfgSection('preprocess')
    return self.preprocess_info
  
  def add_network(self, section_name:str):
    self.layer_info_list.append(NpxCfgSection(section_name))
    return self.layer_info_list[-1]