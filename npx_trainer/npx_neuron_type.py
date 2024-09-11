from collections import namedtuple

from torch import Tensor

QTensor = namedtuple('QTensor', ['tensor', 'dqfactor', 'zero_point'])

class NpxNeuronType():
  def __init__(self, type_by_str:str=None):
    assert type_by_str!=None, type_by_str
    assert type_by_str[0]=='q', type_by_str
    
    non_decimal_index = 1
    while 1:
      if not type_by_str[non_decimal_index].isdecimal():
        break
      non_decimal_index += 1
      
    bit_str = type_by_str[1:non_decimal_index]
    feature_str = type_by_str[non_decimal_index:]
    
    self.num_bits = int(bit_str)
    default_feature_str = 'ssf'
    feature_str = feature_str + default_feature_str[len(feature_str):]
    if feature_str[0]=='s':
      self.is_signed_weight = True
    elif feature_str[0]=='u':
      self.is_signed_weight = False
    else:
      assert 0
    if feature_str[1]=='s':
      self.is_signed_potential = True
    elif feature_str[1]=='u':
      self.is_signed_potential = False
    else:
      assert 0
    
    if feature_str[2]=='f':
      self.is_infinite_potential = False
      self.is_ftarget_updatable  = False
    elif feature_str[2]=='i':
      self.is_infinite_potential = True
      self.is_ftarget_updatable  = False
    elif feature_str[2]=='u':
      self.is_infinite_potential = True
      self.is_ftarget_updatable  = True
    else:
      assert 0
    
    self.mapped_fvalue = 1.0
    
  def update_mapped_fvalue(self, fvalue:float):
    if self.is_infinite_potential:
      assert fvalue > 0, fvalue
      self.mapped_fvalue = fvalue
            
  def __repr__(self):
    result = (self.num_bits, self.is_signed_weight, self.is_signed_potential, self.is_infinite_potential)
    return str(result)
          
  @property
  def umax(self):
    return int(2.**(self.num_bits)) - 1
  
  @property
  def umin(self):
    return 0
        
  @property
  def smax(self):
    return int(2.**(self.num_bits-1)) - 1
    
  @property
  def smin(self):
    return -int(2.**(self.num_bits-1))
  
  @property
  def qscale(self):
    return int(2.**(self.num_bits-1)) if self.is_signed_weight else int(2.**(self.num_bits))
  
  @property
  def qmax(self):
    return self.qscale-1

  @property
  def qmin(self):
    return (self.smin+1) if self.is_signed_weight else 0
  
  @property
  def qfactor(self):
    return float(self.qscale)/self.mapped_fvalue
  
  @property
  def qfactor(self):
    return float(self.qscale)/self.mapped_fvalue
  
  @property
  def dqfactor(self):
    return self.mapped_fvalue / self.qscale
  
  def can_learn_threshold(self):
    return False
  
  def update_ftarget(self, x:Tensor):
    if self.is_ftarget_updatable:
      self.mapped_fvalue = float(x.abs().max())

  def quantize_tensor(self, x:Tensor, bounded:bool):
    self.update_ftarget(x)
    qx = x*self.qfactor
    if bounded:
      qx.clamp_(self.qmin, self.qmax)
    qx.round_()
    return QTensor(qx, self.dqfactor, 0)
  
  def clamp_weight_(self, x:Tensor, is_quantized:bool):
    if not self.is_signed_weight:
      x.clamp_(min=0)
  
  def clamp_mem_(self, x:Tensor, is_quantized:bool):
    if not self.is_signed_potential:
      x.clamp_(min=0)

  @staticmethod
  def dequantize_tensor(qx:QTensor):
    return qx.tensor.float()*qx.dqfactor