import torch
from config import cfg

from model import make_model

if __name__ == "__main__":
    cfg.merge_from_file("configs/Market/pit_transreid_stride.yml")
    cfg.freeze()
    model = make_model(cfg, num_class=3, camera_num=1, view_num=1)
    x = torch.randn([1, 3, 224, 224])
    y = model(x)
    print(y.shape)