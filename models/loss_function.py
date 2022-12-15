# Dice loss function
import torch


def load_loss_fn(opt):
    if opt.loss_function == 'dice_loss':
        return dice_loss
        
    raise Exception('No loss function is loaded.')

def dice_loss(inputs, targets):
    num = inputs * targets
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)

    den1 = inputs * inputs
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)

    den2 = targets * targets
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)

    dice = 2 * (num / (den1 + den2 + 1e-4))

    dice_total = 1 - 1 * torch.sum(dice) / dice.size(0)  # divide by batchsize

    return dice_total
