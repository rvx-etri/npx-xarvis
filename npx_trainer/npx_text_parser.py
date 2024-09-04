from pathlib import *
import math
import copy
import os

from collections import OrderedDict
import re

class NpxTextParser():
  def __init__(self, path:Path):
    super().__init__()

    self.path = path
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
  
  def parsing(self):
    if self.path.is_file():
      file = open(self.path, 'r')
      lines = file.readlines()
      file.close()
      # print(lines)

      for line in lines:
        if '[' in line and ']' in line:
          section_name = self.find_section_name(line)[0]
          current_option_list = OrderedDict([('section', section_name)])
          self.section_list.append(current_option_list)
        elif line.startswith('\0') or line.startswith('#') or line.startswith(';') or line.startswith('\n'):
            pass
        else:
            current_option_list.update(self.get_option(line))
    else:
      print(f'{self.path} file is not exist!')
      assert 0, self.path

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

  def save(self):
    # print('save', self.path)
    with open(self.path, 'w') as file:
      for section in self.section_list:
        for key in section.keys():
          if key == 'section':
            line = f'[{section[key]}]'
          else:
            line = f'{key}={section[key]}'
          file.write(line+'\n')

  def print(self):
    for section in self.section_list:
      for key in section.keys():
        if key == 'section':
          print(f'[{section[key]}]')
        else:
          print(f'{key}={section[key]}')
  
  def add_section(self, section_name:str):
    self.section_list.append(OrderedDict([('section', section_name)]))

  def add_option(self, section_id:int, key:str, value:str):
    if (section_id < len(self.section_list)) & (section_id >= -len(self.section_list)):
      self.section_list[section_id].update({key: str(value)})
    else: 
      print(f'section_id: {section_id} is out of i range for section list. ( < {len(self.section_list)}])')
      assert 0, section_id