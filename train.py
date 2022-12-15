import math
import time
import torch
from options.train_options import TrainOptions
from models.model import load_model
from utils.util import load_results, load_directory, save_images, save_best_model, save_model, update_result, save_results
from data.load_data import dl
from models.loss_function import load_loss_fn
from eval.eval import f1
import math
from utils.visualize import plot
from data.ricap import load_ricap
import random
import torchvision


opt = TrainOptions().get_options()  # Load options
checkpoint_directory, result_directory = load_directory(opt)  # Load directory for checkpoint and result
loss_fn = load_loss_fn(opt)
opt, model, optimizer, scheduler = load_model(opt, checkpoint_directory)  # Load model
results = load_results(opt, result_directory)  # Load result data for loss, accuracy, ...
normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
# Track best model score
best_f1 = 0
best_loss = math.inf
eps = 0  # Counter for early stopping

for epoch in range(opt.last_epoch, opt.epochs):
    epoch += 1
    print('Epoch: {}/{}'.format(epoch, opt.epochs))

    # Load images
    train_dl, valid_dl = dl(opt)

    for phase in opt.phase:
        step = 0  # Record step taken for an epochs
        # Record loss, accuracy... per batch
        running_loss, running_acc, running_f1, running_prec, running_recall = 0, 0, 0, 0, 0

        # Train or valid phase
        if phase == 'train':
            model.train(True)
            dataloader = train_dl
        else:
            model.train(False)
            dataloader = valid_dl

        # Load data x-input image, y-ground truth, z-dilated ground truth
        for x, y, z in dataloader:
            step += 1
            x = normalize(x)  # normalize input image
            
            if random.random() < opt.ricap_probability:
                x, y, z = load_ricap(x, y, z)
            
            
            if torch.cuda.is_available():
                x, y, z = x.cuda(), y.cuda(), z.cuda()

            # Train if phase=train
            if phase == 'train':
                # prediction image
                optimizer.zero_grad()
                pred = model(x)
                pred = torch.sigmoid(pred)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

            # Validation if phase=valid
            elif phase == 'valid':
                with torch.no_grad():
                    pred = model(x)
                    pred = torch.sigmoid(pred)
                    loss = loss_fn(pred, y)

            # Output image binary result with IoU confident threshold 0.5
            threshold_pred = pred > 0.5
            threshold_pred = threshold_pred.float()

            # Save images during validation phase
            if epoch % opt.save_result_images_frequency == 1 or epoch == opt.epochs:
                if phase == 'valid':
                    images = {'input_image': x, 'ground_truth': y, 'predicted_image': threshold_pred}
                    save_images(opt, result_directory, images, epoch, step)

            # Calculate result accuracy, f1 score...
            acc, prec, recall, f1score = f1(threshold_pred, y, z)

            # Record total result accuracy, loss... per epoch
            batch_size = len(x)  # Batch size at current step
            running_acc += acc*batch_size
            running_loss += loss.cpu().detach().numpy()*batch_size
            running_f1 += f1score*batch_size
            running_prec += prec*batch_size
            running_recall += recall*batch_size
            print('---Current step: {}  Loss: {:.4f}  Acc: {:.4f}  F1 Score: {:.4f}  AllocMem (Mb): {:.4f}'.format(step,
                                                                                                                   loss,
                                                                                                                   acc,
                                                                                                                   f1score,
                                                                                                                   torch.cuda.memory_allocated() / 1024 / 1024))
        # Calculate average epoch results
        datasize = len(dataloader.dataset)  # Size of total images used for training
        epoch_loss = round(running_loss / datasize, 4)
        epoch_acc = round(running_acc / datasize, 4)
        epoch_f1 = round(running_f1 / datasize, 4)
        epoch_prec = round(running_prec / datasize, 4)
        epoch_recall = round(running_recall / datasize, 4)

        # Save result loss, accuracy...
        results = update_result(results, phase, epoch_loss, epoch_acc, epoch_f1, epoch_prec, epoch_recall)
        save_results(opt, result_directory, results)
        
        # Print result
        print('Phase: {} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1))
        print('----' * 20)

        # Save best model
        if phase == 'valid' and epoch_f1 > best_f1:
            save_best_model(opt, checkpoint_directory, model)

        # Early stop
        if phase == 'valid' and epoch_loss > best_loss:
            eps += 1
            if eps > 9:
                print(f'Early Stopping at epoch {epoch}, with best loss: {best_loss}, and best f1 score: {best_f1}')
                # Save the last model before early stopping
                save_model(opt, checkpoint_directory, epoch, model, optimizer, scheduler, finish_training=True)
                break
        elif phase == 'valid' and epoch_loss < best_loss:
            best_loss = epoch_loss
            eps = 0

    # Save model at the end of epoch
    if epoch % opt.save_model_frequency == 0:
        finish_training = False
        if epoch == opt.epochs:
            finish_training = True
        save_model(opt, checkpoint_directory, epoch, model, optimizer, scheduler,
                   finish_training=finish_training)

# Plot graph
plot(results, result_directory)
