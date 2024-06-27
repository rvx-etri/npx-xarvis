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
  def qmax(self):
    return (self.smax + 1) if self.is_signed_weight else (self.umax + 1)

  @property
  def qmin(self):
    return self.smin if self.is_signed_weight else 0

  def quantize_tensor(self, x:Tensor, bounded:bool=True):
    fval_max = 1.0

    scale = fval_max / self.qmax
    
    qx = x*self.qmax/fval_max # x/scale

    if bounded:
      qx.clamp_(self.qmin, self.qmax-1)
    qx.round_()
    return QTensor(qx, scale, 0)

  def dequantize_tensor(self, qx:QTensor):
    return qx.scale * (qx.tensor.float())
