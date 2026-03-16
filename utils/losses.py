import torch
import torch.nn.functional as F


def charbonnier_loss(pred,target,eps=1e-3):

    return torch.mean(torch.sqrt((pred-target)**2+eps**2))


def combined_loss(pred,target):

    return 0.5*F.mse_loss(pred,target)+0.5*charbonnier_loss(pred,target)
