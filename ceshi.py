import torch
import timm

from model.backbones.pit import pit_b
m = pit_b(True)
print(m)