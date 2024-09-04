from collections import namedtuple

from torch import Tensor

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

class NpxNeuronType():
  def __init__(self, type_by_str:str=None):
    assert type_by_str!=None, type_by_str
    assert type_by_str[0]=='q', type_by_str
    
    self.num_bits = int(type_by_str[1])
    if type_by_str[2]=='s':
      self.is_signed_weight = True
    elif type_by_str[2]=='u':
      self.is_signed_weight = False
    else:
      assert 0
    if type_by_str[3]=='s':
      self.is_signed_potential = True
    elif type_by_str[3]=='u':
      self.is_signed_potential = False
    else:
      assert 0
            
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

  def quantize_tensor(self, x:Tensor, bounded:bool):
    fval_max = 1.0

    scale = fval_max / self.qscale
    
    qx = x*self.qscale/fval_max # x/scale

    if bounded:
      qx.clamp_(self.qmin, self.qmax)
    qx.round_()
    return QTensor(qx, scale, 0)
  
  def clamp_weight_(self, x:Tensor, is_quantized:bool):
    if not self.is_signed_weight:
      x.clamp_(min=0)
  
  def clamp_mem_(self, x:Tensor, is_quantized:bool):
    if not self.is_signed_potential:
      x.clamp_(min=0)

  @staticmethod
  def dequantize_tensor(qx:QTensor):
    return qx.scale * (qx.tensor.float())