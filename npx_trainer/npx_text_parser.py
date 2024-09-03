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

  def find_option_value(self, option_list:OrderedDict, key:str, default:str):
    if option_list.get(key):
      return option_list[key]
    else:
      return default

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
      self.section_list[section_id].update({key: value})
    else: 
      print(f'section_id: {section_id} is out of i range for section list. ( < {len(self.section_list)}])')
      assert 0, section_id