from pathlib import *
import math
import copy
import os

import re

class NpxTextParser():
  def __init__(self, path:Path):
    super(NpxTextParser, self).__init__()

    self.path = path
    self.find_section_name = re.compile('\[([^]]+)\]').findall
    self.section_list = []

    if self.path.is_file():
      self.parsing()
    else:
      print(f'{self.path} file is not exist!')
      assert 0, self.path

  def get_option(self, line):
    strings = line.strip().split('=')
    return {strings[0]: strings[1]}
  
  def parsing(self):
    file = open(self.path, 'r')
    lines = file.readlines()
    file.close()
    # print(lines)

    for line in lines:
      if '[' in line and ']' in line:
        section_name = self.find_section_name(line)[0]
        current_option_list = dict([('section', section_name)])
        self.section_list.append(current_option_list)
      elif line.startswith('\0') or line.startswith('#') or line.startswith(';') or line.startswith('\n'):
          pass
      else:
          current_option_list.update(self.get_option(line))

  def find_option_value(self, option_list, key, default):
    if option_list.get(key):
      return option_list[key]
    else:
      return default