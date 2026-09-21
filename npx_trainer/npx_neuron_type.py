from collections import namedtuple
import math
import re

import torch
from torch import Tensor

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

_warned_keys = set()

def warn_once(key, message:str):
  if key in _warned_keys:
    return
  _warned_keys.add(key)
  print(f'[warning] {message}')

def to_dtype_toward_zero(value:float, dtype:torch.dtype):
  result = torch.tensor(value, dtype=dtype)
  if abs(result.item()) > abs(value):
    result = torch.nextafter(result, torch.zeros((), dtype=dtype))
  return result.item()

_NEW_FORMAT = re.compile(r'w(f|[su]\d+)(?:-m(f|[su]\d+))?')
_OLD_FORMAT = re.compile(r'([qc])(\d+)([su])([su])([fiu])')

MAX_WEIGHT_BITS = 16
FLOAT_WEIGHT_BITS = 32
MAX_MEM_BITS = 32
DEFAULT_OVERFLOW_HEADROOM = 0.5

def convert_old_neuron_type(type_by_str:str):
  match = _OLD_FORMAT.fullmatch(type_by_str)
  if not match:
    raise ValueError(f'invalid neuron_type {type_by_str!r}')
  datatype, bit_str, weight_sign, potential_sign, scale_mode = match.groups()
  result = 'wf' if datatype=='c' else f'w{weight_sign}{bit_str}'
  if (potential_sign=='u') and (datatype!='c'):
    result += f'-mu{MAX_MEM_BITS}'
  message = f'neuron_type {type_by_str!r} is deprecated, use {result!r}'
  if (potential_sign=='u') and (datatype=='c'):
    message += '; the unsigned potential is dropped, a float membrane (mf) has no sign option'
  if scale_mode!='u':
    message += (f'; the fixed/threshold scale mode {scale_mode!r} no longer exists,'
                ' the scale is now derived from the weight max (see q_max)')
  warn_once(('neuron_type', type_by_str), message)
  return result

class NpxNeuronType():
  def __init__(self, type_by_str:str=None):
    assert type_by_str!=None, type_by_str
    type_by_str = str(type_by_str)
    if type_by_str[:1] in ('q', 'c'):
      type_by_str = convert_old_neuron_type(type_by_str)

    match = _NEW_FORMAT.fullmatch(type_by_str)
    if not match:
      raise ValueError(f'invalid neuron_type {type_by_str!r}: expected w<s|u><bits>[-m<s|u><bits>] or wf[-mf]')
    weight_str, mem_str = match.groups()

    self.is_quantized = (weight_str!='f')
    if mem_str is None:
      mem_str = f's{MAX_MEM_BITS}' if self.is_quantized else 'f'
    if (mem_str!='f') != self.is_quantized:
      raise ValueError(f'invalid neuron_type {type_by_str!r}: weight and membrane must be both quantized or both float (wf-mf)')

    if self.is_quantized:
      self.num_bits = int(weight_str[1:])
      self.is_signed_weight = (weight_str[0]=='s')
      min_bits = 2 if self.is_signed_weight else 1
      if not (min_bits <= self.num_bits <= MAX_WEIGHT_BITS):
        raise ValueError(f'invalid neuron_type {type_by_str!r}: weight bits must be {min_bits}..{MAX_WEIGHT_BITS}')
      self.mem_bits = int(mem_str[1:])
      self.is_signed_potential = (mem_str[0]=='s')
      if not (2 <= self.mem_bits <= MAX_MEM_BITS):
        raise ValueError(f'invalid neuron_type {type_by_str!r}: membrane bits must be 2..{MAX_MEM_BITS}')
    else:
      self.num_bits = FLOAT_WEIGHT_BITS
      self.is_signed_weight = True
      self.mem_bits = MAX_MEM_BITS
      self.is_signed_potential = True

    self.reset_mechanism = 'subtract'
    self.overflow_headroom = DEFAULT_OVERFLOW_HEADROOM
    self.input_depth = 1
    self.q_max_source = None
    self._q_max = None

  def configure_membrane(self, reset_mechanism:str, overflow_headroom:float=DEFAULT_OVERFLOW_HEADROOM):
    try:
      overflow_headroom = float(overflow_headroom)
    except (TypeError, ValueError):
      raise ValueError(f'overflow_headroom must be a number in [0, 1), got {overflow_headroom!r}')
    if not (0. <= overflow_headroom < 1.):
      raise ValueError(f'overflow_headroom must be in [0, 1), got {overflow_headroom}')
    self.reset_mechanism = reset_mechanism
    self.overflow_headroom = overflow_headroom
    self.threshold_qcode_max

  def __repr__(self):
    result = (self.num_bits, self.is_signed_weight, self.mem_bits, self.is_signed_potential, self.is_quantized)
    return str(result)

  @property
  def name(self):
    if not self.is_quantized:
      return 'wf'
    result = 'w' + ('s' if self.is_signed_weight else 'u') + str(self.num_bits)
    if (self.mem_bits < MAX_MEM_BITS) or (not self.is_signed_potential):
      result += '-m' + ('s' if self.is_signed_potential else 'u') + str(self.mem_bits)
    return result

  @property
  def full_name(self):
    if not self.is_quantized:
      return 'wf-mf'
    return ('w' + ('s' if self.is_signed_weight else 'u') + str(self.num_bits)
            + '-m' + ('s' if self.is_signed_potential else 'u') + str(self.mem_bits))

  @property
  def qcode_max(self):
    if self.is_signed_weight:
      return int(2**(self.num_bits-1)) - 1
    return int(2**self.num_bits) - 1

  @property
  def qcode_min(self):
    if self.is_signed_weight:
      return -self.qcode_max
    return 0

  @property
  def mem_qcode_max(self):
    if self.is_signed_potential:
      return int(2**(self.mem_bits-1)) - 1
    return int(2**self.mem_bits) - 1

  @property
  def mem_qcode_min(self):
    if self.is_signed_potential:
      return -int(2**(self.mem_bits-1))
    return 0

  @property
  def threshold_qcode_max(self):
    limit = self.mem_qcode_max - 1
    if self.reset_mechanism=='subtract':
      limit = math.floor(limit*(1.-self.overflow_headroom))
    if limit < 1:
      raise ValueError(f'{self.name}: overflow_headroom={self.overflow_headroom} leaves no threshold code in the membrane register')
    return limit

  def update_q_max(self, weight_list, threshold):
    if not self.is_quantized:
      return
    if self.q_max_source is not None:
      self._q_max = self.q_max_source.q_max*self.qcode_max/self.q_max_source.qcode_max
      return
    q_max = 0.
    for weight in weight_list:
      weight = weight.detach().abs() if self.is_signed_weight else weight.detach().clamp(min=0)
      q_max = max(q_max, float(weight.max()))
    threshold = float(torch.as_tensor(threshold).detach().abs().max())
    if self.input_depth == 1:
      q_max = max(q_max, self.qcode_max*threshold/self.threshold_qcode_max)
    elif self.input_depth > 1:
      q_max = max(q_max, self.qcode_max*(threshold/self.threshold_qcode_max)**(1./self.input_depth))
    if q_max > 0:
      self._q_max = q_max
    elif self._q_max is None:
      raise ValueError(f'{self.name}: cannot derive q_max, no weight is above zero, the threshold is zero and there is no previous value')

  @property
  def q_max(self):
    if self._q_max is None:
      raise RuntimeError(f'{self.name}: q_max is not derived yet, call update_q_max() first')
    return self._q_max

  @property
  def inv_scale(self):
    return float(self.qcode_max)/self.q_max

  @property
  def scale(self):
    return self.q_max / self.qcode_max

  def quantize_tensor(self, x:Tensor, bounded:bool):
    if not self.is_quantized:
      return QTensor(x.clone(), 1.0, 0)
    qcode = x*self.inv_scale
    if bounded:
      qcode.clamp_(self.qcode_min, self.qcode_max)
    qcode.round_()
    return QTensor(qcode, self.scale, 0)

  def quantize_threshold(self, threshold:Tensor, input_factor:float):
    if not self.is_quantized:
      return QTensor(threshold.clone(), 1.0, 0)
    qcode = threshold*input_factor
    qcode.round_()
    qcode.clamp_(max=to_dtype_toward_zero(self.threshold_qcode_max, qcode.dtype))
    return QTensor(qcode, 1.0/input_factor, 0)

  def clamp_weight_(self, x:Tensor, is_quantized:bool):
    if not self.is_signed_weight:
      x.clamp_(min=0)

  def clamp_mem_(self, x:Tensor, is_quantized:bool):
    if not self.is_quantized:
      return
    scale = 1.0 if is_quantized else self.scale**self.input_depth
    x.clamp_(to_dtype_toward_zero(self.mem_qcode_min*scale, x.dtype),
             to_dtype_toward_zero(self.mem_qcode_max*scale, x.dtype))

  def quantize_beta(self, beta:float):
    denominator = 256
    beta_numerator = int(beta*denominator)
    result = float(beta_numerator) / denominator
    return result

  @staticmethod
  def dequantize_tensor(qtensor:QTensor):
    return qtensor.tensor.float()*qtensor.scale
