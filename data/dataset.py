from pathlib import Path
from os.path import splitext
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from os import listdir
from os.path import splitext


# Dataset
class BasicDataset(Dataset):
    def __init__(self, img_dir, label_dir, dilated_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        if dilated_dir is not None:
            self.dilated_dir = Path(dilated_dir)
        self.transform = transform
        self.ids = [splitext(file)[0] for file in listdir(self.img_dir) if
                    not file.startswith('.')]  # Extract filenames in folder
        self.TOTENSOR = ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]  # Image path name
        image_path = list(self.img_dir.glob(name + '.*'))[0]  # Image path
        image = Image.open(image_path)

        suffix = '_'.join(name.split('_')[1:])
        label_name = '_'.join(('overlapped', suffix))
        label_path = list(self.label_dir.glob(label_name + '.*'))[0]
        label = Image.open(label_path)
        label = self.TOTENSOR(label)
        
        # Transform/Augment
        if self.transform is not None:
            image = self.transform(image)

        if self.dilated_dir is not None:
            dilated_name = '_'.join(('overlapped', suffix))
            dilated_path = list(self.dilated_dir.glob(dilated_name + '.*'))[0]
            dilated = Image.open(dilated_path)
            dilated = self.TOTENSOR(dilated)
            if self.transform is None:
                image = self.TOTENSOR(image)
            return image, label, dilated
        return image, label
