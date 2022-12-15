import matplotlib.pyplot as plt
import os
import csv
import torchvision.transforms.functional as F
import numpy as np


def plot(results, save_directory):

    variable2plot = [v for v in results['train']]
    for variable in variable2plot:
        plt.figure()

        plt.xlabel('Epoch')  # label x axis
        handles = []
        for phase in results:
            x = range(1, len(results[phase][variable]) + 1)
            plt.ylabel(variable) # label y axis
            line, = plt.plot(x, results[phase][variable], label=phase)
            handles.append(line)

        plt.legend(handles=handles)

    # Save figure
        if os.path.isdir(save_directory):
            graph_dir = os.path.join(save_directory, 'graph')
            save_path = os.path.join(graph_dir, variable + '.jpeg')
            if os.path.isdir(graph_dir):
                plt.savefig(save_path)
            else:
                os.mkdir(graph_dir)
                plt.savefig(save_path)


# Save data to excel function
def save2csv(datas, title, dir=None):
    directory = os.path.join(dir, str(title) + '.csv')

    with open(directory, 'w') as f:
        writer = csv.writer(f)

        for data in datas:
            writer.writerow(data)


# Read Excel data
def read_csv(title, dir=None):
    directory = os.path.join(dir,str(title)+'.csv')
    store = []
    with open(directory, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row != []:
                row = [float(x) for x in row]
                print(row)
                store.append(row)

    epochs = store[0]
    training = store[1]
    valid = store[2]
    return epochs, training, valid

def show_image(imgs):   
  if not isinstance(imgs, list):
        imgs = [imgs]
        
  fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
   
  for i,img in enumerate(imgs):
        imag = img[0].detach()
        img = F.to_pil_image(imag)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        