import numpy as np
import torch
import random
from eval.eval import f1
import pdb


def load_ricap(images, targets, dilates):
    batch_images = torch.empty(images.size())
    batch_targets = torch.empty(targets.size())
    batch_dilates = torch.empty(dilates.size())
    for i in range(images.size()[0]):
        patched_image, patched_target, patched_dilate = ricap(images, targets, dilates)
        batch_images[i] = patched_image
        batch_targets[i] = patched_target
        batch_dilates[i] = patched_dilate

    return batch_images, batch_targets, batch_dilates


def ricap(images, targets, dilates):

    # size of image
    I_x, I_y = images.size()[2:]

    # generate boundary position (w, h)
    w = int(np.round(I_x * np.random.uniform(0.3, 0.7)))
    h = int(np.round(I_y * np.random.uniform(0.3, 0.7)))
    w_ = [w, I_x-w, w, I_x-w]
    h_ = [h, h, I_y-h, I_y-h]

    # select four images
    cropped_images = {}
    cropped_target = {}
    cropped_dilate = {}
    c_ = {}
    W_ = {}

    for k in range(4):
        index = random.choice(torch.randperm(images.size(0)))
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_y - h_[k] + 1)

        cropped_images[k] = images[index, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        cropped_target[k] = targets[index, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        cropped_dilate[k] = dilates[index, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        c_[k] = targets[index]
        W_[k] = (w_[k] * h_[k]) / (I_x * I_y)

    # patch cropped images
    patched_images = torch.cat(
        (torch.cat((cropped_images[0], cropped_images[1]), 1),
         torch.cat((cropped_images[2], cropped_images[3]), 1)),
        2)
    patched_target = torch.cat(
        (torch.cat((cropped_target[0], cropped_target[1]), 1),
         torch.cat((cropped_target[2], cropped_target[3]), 1)),
        2)
    print(patched_target.size())
    patched_dilate = torch.cat(
        (torch.cat((cropped_dilate[0], cropped_dilate[1]), 1),
         torch.cat((cropped_dilate[2], cropped_dilate[3]), 1)),
        2)

    targets = (c_, W_)
    return patched_images, patched_target, patched_dilate


if __name__ == '__main__':
    input_image = torch.rand(4, 3, 256, 256)
    input_target = torch.rand(4, 1, 256, 256)
    input_target = (input_target> 0.5)*1

    image, target, dilate = load_ricap(input_image, input_target, input_target)
    print(target.size())

    acc, _, _, f1_score = f1(target, target, dilate)
    print(acc, f1_score)
