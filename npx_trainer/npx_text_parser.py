from pathlib import *
import math
import copy
import os

from collections import OrderedDict
import re

class NpxTextParser():
  def __init__(self):
    super().__init__()
    self.find_section_name = re.compile('\[([^]]+)\]').findall
    self.section_list = []
    
  @property
  def global_info(self):
    return self.section_list[0]
  
  @property
  def layer_info_list(self):
    return self.section_list[1:]

  def get_option(self, line:str):
    strings = line.strip().split('=')
    return {strings[0]: strings[1]}
  
  def parse_file(self, path:Path):
    assert path.is_file(), path
    line_list = path.read_text().split('\n')

    for line in line_list:
      if line=='':
        continue
      if '[' in line and ']' in line:
        section_name = self.find_section_name(line)[0]
        current_option_list = OrderedDict([('section', section_name)])
        self.section_list.append(current_option_list)
      elif line.startswith('\0') or line.startswith('#') or line.startswith(';') or line.startswith('\n'):
          pass
      else:
        current_option_list.update(self.get_option(line))

  @staticmethod
  def find_option_value(option_list:OrderedDict, key:str, default_value):
    if option_list.get(key):
      assert default_value!=None, key
      value = option_list[key]
      if type(default_value)==type(True):
        if value=='True':
          value = True
        elif value=='False':
          value = False
        else:
          assert 0, value
      else:
        value = type(default_value)(value)
    else:
      value = default_value
    return value
  
  def __str__(self) -> str:
    line_list = []
    for section in self.section_list:
      for key in section.keys():
        if key == 'section':
          line = f'[{section[key]}]'
        else:
          line = f'{key}={section[key]}'
        line_list.append(line)
    return '\n'.join(line_list)
  
  def __repr__(self) -> str:
    return str(self)+'\n' 
  
  def write_file(self, path:Path):
    path.write_text(str(self))
    
  def add_section(self, section_name:str):
    self.section_list.append(OrderedDict([('section', section_name)]))

  def add_option(self, section_id:int, key:str, value:str):
    if (section_id < len(self.section_list)) & (section_id >= -len(self.section_list)):
      self.section_list[section_id].update({key: str(value)})
    else: 
      print(f'section_id: {section_id} is out of i range for section list. ( < {len(self.section_list)}])')
      assert 0, section_id

  def del_option(self, section_id:int, key:str):
    if (section_id < len(self.section_list)) & (section_id >= -len(self.section_list)):
      self.section_list[section_id].pop(key, None)
    else: 
      print(f'section_id: {section_id} is out of i range for section list. ( < {len(self.section_list)}])')
      assert 0, section_id
