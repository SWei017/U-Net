import os

import torch
import os.path as osp
import pickle
from torchvision.utils import save_image


def make_directory(directory):
    for i, x in enumerate(directory.split('/')):
        if i != len(directory.split('/'))-2:
            pth = '/'.join(directory.split('/')[:i+2])
            pth = osp.join(pth)
            if not osp.isdir(pth):
                os.mkdir(pth)


def check_previous_training(opt):
    # Return True if to continue last training
    checkpoint_directory = osp.join(opt.model_weight_save_path, opt.input_image_directory+' '+opt.optimizer+' '+
                                    opt.loss_function+opt.filename_suffix)
    if osp.isfile(osp.join(checkpoint_directory, 'model_state_dict.pth')):
        opt.cont_train = True
        print('Previous model detected')
    return opt


# Search and load model
def load_directory(opt):
    # cont_train input as True or False
    # Return checkpoint and result directory
    # Make directory if not exist
    checkpoint_directory = osp.join(opt.model_weight_save_path, opt.input_image_directory+' '+
                                    opt.optimizer+' '+opt.loss_function+opt.filename_suffix)
    result_directory = osp.join(opt.result_saving_dir, opt.input_image_directory+' '+ opt.optimizer+' '+opt.loss_function+opt.filename_suffix)
    if not opt.cont_train and not osp.isdir(checkpoint_directory):
        make_directory(checkpoint_directory)
    if not opt.cont_train and not osp.isdir(result_directory):
        make_directory(result_directory)
    return checkpoint_directory, result_directory


# Save best model state dict
def save_best_model(opt, checkpoint_directory, model):
    # Only save model state dict
    # ie, './checkpoints/Unet/overlapped/Adamax/dice/best_model_state_dict.pth
    saving_path = osp.join(checkpoint_directory,opt.best_model_state_dict)
    torch.save({
        'model_state_dict': model.state_dict(),
    }, saving_path)


# Save model from running epochs
def save_model(opt, checkpoint_directory, epoch, model, optimizer, scheduler, finish_training=False):
    # Specify path
    # finish_trained - True if model done training
    # ie, './checkpoints/Unet/overlapped/Adamax/dice/model_state_dict.pth
    saving_path = osp.join(checkpoint_directory,opt.model_state_dict)
    torch.save({
        'last_epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'finish_training': finish_training
    }, saving_path)


# Save loss, accuracy, ... results from training or testing model
def save_results(opt, result_directory, result_dicts):
    # Saving path for data, ie loss, accuracy...
    # ie, './results/Unet/overlapped/Adamax/dice/result_dicts
    file_path = osp.join(result_directory, opt.result_data_filename)
    with open(file_path, 'wb') as f:
        pickle.dump(result_dicts, f)


def load_results(opt, result_directory):
    if opt.cont_train:
        file_path = osp.join(result_directory, opt.result_data_filename)
        with open(file_path, 'rb') as f:
            loaded_dict = pickle.load(f)
    else:
        loaded_dict = {'train': {'loss': [],
                                    'accuracy': [],
                                    'f1_score': [],
                                    'precision': [],
                                    'recall': []},
                       'valid': {'loss': [],
                                 'accuracy': [],
                                 'f1_score': [],
                                 'precision': [],
                                 'recall': []}}
    return loaded_dict


# Update result
def update_result(result_dict, phase, loss, accuracy, f1score, precision, recall):
    result_dict[phase]['loss'].append(loss)
    result_dict[phase]['accuracy'].append(accuracy)
    result_dict[phase]['f1_score'].append(f1score)
    result_dict[phase]['precision'].append(precision)
    result_dict[phase]['recall'].append(recall)
    return result_dict


def save_images(opt, directory, images_dict, epoch, step):
    # Save result training or testing images
    # images - images in dictionary, ie images = {'input_image': [...]}
    for image in images_dict:
        filename = image+'_'+str(epoch)+'_'+str(step)+'.png'
        saving_directory = osp.join(directory, opt.result_image_folder)
        if not os.path.isdir(saving_directory):
            os.mkdir(saving_directory)
        save_filename = osp.join(saving_directory, filename)
        save_image(images_dict[image], save_filename, pad_value=1)



