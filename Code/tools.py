import torch.nn.init as init
import torch.nn as nn

import numpy as np
import cv2


# TODO set the argument for diiferent options on initialization
def initialize_weights(method='xavier', *models):
    for model in models:
        for module in model.modules():

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                if method == 'kaiming':
                    init.kaiming_normal(module.weight.data, np.sqrt(2.0))
                elif method == 'xavier':
                    init.xavier_normal(module.weight.data, np.sqrt(2.0))
                elif method == 'orthogonal':
                    init.orthogonal(module.weight.data, np.sqrt(2.0))
                elif method == 'normal':
                    init.normal(module.weight.data, mean=0, std=0.02)
                if module.bias is not None:
                    init.constant(module.bias.data, 0)

def dice_loss (pred,target,smooth=1.):
  pred = pred.contiguous()
  target = target.contiguous()    

  intersection = (pred * target).sum(dim=2).sum(dim=2)
  union=  (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2))-intersection
  jacc=intersection/union
  jacc_loss=1-jacc
  dice_loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
  dice=1-dice_loss  
  return dice_loss.mean(),jacc_loss.mean(),jacc.mean(),dice.mean()


def loss_function(bce_loss,dice_loss,bce_weight):
  loss = bce_loss * bce_weight + dice_loss * (1 - bce_weight)
  return loss