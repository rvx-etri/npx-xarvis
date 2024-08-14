import os
import argparse
from pathlib import *

from npx_define import *
  
def analyze_best_result(npx_define:NpxDefine):
  line_list = []
  for report_path in npx_define.report_dir_path.glob(npx_define.get_report_filename_pattern()):
    line_list += report_path.read_text().split('\n')
    
  best_val_accuracy = 0
  best_result = None
  for line in line_list:
    if not line:
      pass
    single_result = RecordResult(*line.split('|'))
    val_accuracy = float(single_result.val_accuracy_str)
    if val_accuracy > best_val_accuracy:
      best_val_accuracy  = val_accuracy
      best_result = single_result
  return best_result

def copy_best_parameter(npx_define:NpxDefine, best_result:RecordResult):
  best_parameter_path = npx_define.get_parameter_path(int(best_result.repeat_index_str),int(best_result.epoch_index_str),True)
  assert best_parameter_path.is_file(), best_parameter_path
  riscv_parameter_path = npx_define.get_riscv_parameter_path(True)
  npx_define.riscv_dir_path.mkdir(parents=True, exist_ok=True)
  riscv_parameter_path.write_bytes(best_parameter_path.read_bytes())
  
  best_parameter_text_path = npx_define.get_parameter_text_path(int(best_result.repeat_index_str),int(best_result.epoch_index_str),True)
  if best_parameter_text_path.is_file():
    riscv_parameter_text_path = npx_define.get_riscv_parameter_text_path(True)
    npx_define.riscv_dir_path.mkdir(parents=True, exist_ok=True)
    riscv_parameter_text_path.write_bytes(best_parameter_text_path.read_bytes())

def generate_riscv_binary(npx_define:NpxDefine):
  riscv_parameter_path = npx_define.get_riscv_parameter_path(True)
  assert riscv_parameter_path.is_file(), riscv_parameter_path
  
if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='NPX Framework')
  parser.add_argument('-cfg', '-c', nargs='+', help='app cfg file name')
  parser.add_argument('-output', '-o', help='output directory')

  # check args
  args = parser.parse_args()
  assert args.cfg
  assert args.output

  app_cfg_list = args.cfg
  output_path = Path(args.output).absolute()
  assert output_path.is_dir(), output_path
  
  # cfg
  for app_cfg in app_cfg_list:
    app_cfg_path = Path(app_cfg)
    assert app_cfg_path.is_file(), app_cfg_path
    
    npx_define = NpxDefine(app_cfg_path=app_cfg_path, output_path=output_path)    
    best_result = analyze_best_result(npx_define)
    copy_best_parameter(npx_define,best_result)
    generate_riscv_binary(npx_define)
    