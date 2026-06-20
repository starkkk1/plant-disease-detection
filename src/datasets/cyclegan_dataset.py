import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def make_dataset(directory):
    """Recursively finds all images in a directory."""
    images = []
    assert os.path.isdir(directory), '%s is not a valid directory' % directory
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class UnalignedDataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images from domain A '/path/to/data/A'
    and from domain B '/path/to/data/B' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, dir_A, dir_B, image_size=256, is_train=True):
        self.dir_A = dir_A
        self.dir_B = dir_B
        
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        # Standard CycleGAN transforms
        transform_list = [
            transforms.Resize((image_size, image_size), Image.BICUBIC)
        ]
        
        if is_train:
            transform_list.append(transforms.RandomHorizontalFlip())
            
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        # Domain A images are accessed sequentially
        A_path = self.A_paths[index % self.A_size]
        # Domain B images are accessed randomly to break correlation
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        A = self.transform(A_img)
        B = self.transform(B_img)
        
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """As we have two datasets with potentially different number of images,
        we take a maximum of them."""
        return max(self.A_size, self.B_size)
