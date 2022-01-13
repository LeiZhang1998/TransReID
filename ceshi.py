import torch
import timm

# from model.backbones.pit import pit_b
# m = pit_b(True)
# print(m)
from model.make_model import make_model
from config import cfg
cfg.merge_from_file('configs/Market/pit_transreid_stride.yml')
model = make_model(cfg, num_class=3, camera_num=1, view_num = 1)
x = torch.rand([1, 3, 224, 224])
y = model(x)
print(y)