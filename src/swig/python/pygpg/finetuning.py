from pygpg import conversion as C
import torch


def finetune(model):
  torchified = C.expr_to_torch_module(model, torch.float32)
  return model