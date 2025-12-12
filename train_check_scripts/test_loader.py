import torch

ckpt = torch.load("checkpoints/dal/dal-large.pth", map_location="cpu")
print(type(ckpt))
print(ckpt.keys() if isinstance(ckpt, dict) else ckpt)
