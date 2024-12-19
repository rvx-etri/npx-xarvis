import os
import argparse
import time
import shutil
from pathlib import *
from tqdm.auto import tqdm

from npx_define import *
from npx_text_parser import *

def gen_operator_info(npx_define:NpxDefine, hw_info_path:Path):
  hw_info_file = open(hw_info_path, 'r')
  hw_info = hw_info_file.readlines()
  included_hw = []
  for define in hw_info:
    if define.startswith('#define INCLUDE_'):
      included_hw.append(define.strip().replace('#define INCLUDE_', ''))
  #print(included_hw)
    
  operator_info = NpxTextParser()
  for i, layer_option in enumerate(npx_define.text_parser.layer_info_list):
    if layer_option.get('section'):
      operator_info.add_section(layer_option['section'])
      if layer_option['section'] == 'Linear':
        operator_info.add_option(-1, 'operator', 'cpu')
      elif layer_option['section'] == 'Conv2d':
        operator_info.add_option(-1, 'operator', 'cpu')
      elif layer_option['section'] == 'Conv2d':
        operator_info.add_option(-1, 'operator', 'cpu')
      elif layer_option['section'] == 'Conv2d':
        operator_info.add_option(-1, 'operator', 'cpu')
      elif layer_option['section'] == 'Conv2d':
        operator_info.add_option(-1, 'operator', 'cpu')
      elif layer_option['section'] == 'Conv2d':
        operator_info.add_option(-1, 'operator', 'cpu')
      elif layer_option['section'] == 'MaxPool2d':
        operator_info.add_option(-1, 'operator', 'cpu')
      elif layer_option['section'] == 'AvgPool2d':
        operator_info.add_option(-1, 'operator', 'cpu')
      elif layer_option['section'] == 'Flatten':
        operator_info.add_option(-1, 'operator', 'cpu')
      elif layer_option['section'] == 'Leaky':
        operator_info.add_option(-1, 'operator', 'cpu')
      else:
        assert 0

  #print(npx_define.text_parser.layer_info_list)
  #print(str(npx_define.text_parser))
  #print(str(operator_info))

  operator_info_path = npx_define.get_operator_info_path()
  #print(operator_info_path)
  operator_info_path.write_text(str(operator_info))
  
if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='NPX Framework')
  parser.add_argument('-cfg', '-c', nargs='+', help='app cfg file name')
  parser.add_argument('-output', '-o', help='output directory')
  parser.add_argument('-hw_info', '-i', help='hw info directory')

  # check args
  args = parser.parse_args()
  assert args.cfg
  assert args.output
  assert args.hw_info

  app_cfg_list = args.cfg
  #print(app_cfg_list)
  output_path = Path(args.output).absolute()
  hw_info_path = Path(args.hw_info).absolute()

  # cfg
  for app_cfg in app_cfg_list:
    app_cfg_path = Path(app_cfg)
    #print(app_cfg_path)
    npx_define = NpxDefine(app_cfg_path=app_cfg_path, output_path=output_path)
    gen_operator_info(npx_define=npx_define, hw_info_path=hw_info_path)
