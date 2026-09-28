import torch
import snntorch
from torch import Tensor

from npx_neuron_type import NpxNeuronType

class NpxLeaky(snntorch.Leaky):
  def __init__(self, neuron_type:NpxNeuronType, **kwargs):
    super().__init__(**kwargs)
    self.neuron_type = neuron_type
    self.learn_beta = kwargs.get('learn_beta', False)
    self.is_network_quantized = False

  def mem_reset(self, mem:Tensor):
    # snntorch infers "a spike is still waiting to be reset" from mem>threshold,
    # which only holds while mem is kept unreset. Without reset_delay the reset
    # already happened, so a membrane left above threshold made fire() see
    # mem-threshold and drop the spike.
    if not self.reset_delay:
      return torch.zeros_like(mem)
    return super().mem_reset(mem)

  def clamp_mem(self, mem:Tensor):
    # the register saturates as the sum is written into it, so the comparator
    # sees the clamped membrane. The clamp acts on .data, so no gradient reaches
    # the ceiling and training under it does not teach the net to stay below it.
    if self.neuron_type and not self.training:
      self.neuron_type.clamp_mem_(mem, self.is_network_quantized)
    return mem

  def _base_sub(self, input_:Tensor):
    # snntorch subtracts the threshold after the decay, so a reset carried over
    # from the previous step removes a full threshold while the charge it
    # removes has already decayed. Subtracting inside the decay makes the
    # delayed reset equivalent to resetting at the step that spiked.
    return self.clamp_mem(self.beta.clamp(0, 1)*(self.mem - self.reset*self.threshold) + input_)

  # _base_zero (hard reset) is left to snntorch: a membrane above the register is
  # above the threshold too, so the neuron fires and the membrane goes to zero
  def _base_int(self, input_:Tensor):
    return self.clamp_mem(super()._base_int(input_))

  def forward(self, input_:Tensor, mem=None):
    result = super().forward(input_) if mem is None else super().forward(input_, mem)
    if self.neuron_type and self.learn_beta:
      self.beta.data.fill_(self.neuron_type.quantize_beta(self.beta.data.float()))
    return result
