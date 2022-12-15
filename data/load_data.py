from torch.utils.data import DataLoader
from data.dataset import BasicDataset
import torch
import os.path as osp
import torchvision


# Path for orig and segmented images
# Change this directory when run on google colab

# dir_img = '/content/drive/MyDrive/Colab Notebooks/Training & Validating (1)/overlapped images'
# dir_mask = '/content/drive/MyDrive/Colab Notebooks/Training & Validating (1)/labelled images'
# dir_dilated = '/content/drive/MyDrive/Colab Notebooks/Training & Validating (1)/Dilated images'

# transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])


# Function to load image into DataLoader (valid/train)
def dl(opt, transforms=None):
    specific_image_directory = osp.join(opt.img_directory, opt.spec_images_directory)
    dir_img = osp.join(specific_image_directory, opt.input_image_directory)
    dir_mask = osp.join(specific_image_directory, opt.mask_directory)
    dir_dilated = osp.join(specific_image_directory, opt.dilate_directory)
    dataset = BasicDataset(dir_img, dir_mask, dir_dilated, transform=transforms)
    if not opt.isTrain:
        return DataLoader(dataset, batch_size=opt.batch_size)

    training_length = int(0.75 * len(dataset))
    valid_length = int(len(dataset) - training_length)

    valid_data, training_data = torch.utils.data.random_split(dataset, [valid_length, training_length])

    valid_dataloader = DataLoader(valid_data, batch_size=opt.batch_size)
    train_dataloader = DataLoader(training_data, batch_size=opt.batch_size)

    return train_dataloader, valid_dataloader
