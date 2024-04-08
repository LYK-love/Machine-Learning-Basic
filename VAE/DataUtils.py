import os
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, default_label=0, preload=True):
        """
        Args:
            root_dir (string): Base directory for all images.
            train (bool): If True, load images from the 'train' subdirectory. Otherwise, from 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            default_label (int, optional): Default label to use when no labels are found.
            preload (bool, optional): If True, preload all images into memory.
        """
        self.transform = transform
        self.default_label = default_label
        self.preload = preload
        self.data = []
        self.labels = []
        self.images = []

        # Adjust the directory based on the train flag
        data_dir = os.path.join(root_dir, 'train' if train else 'test')
        
        if self.preload:
            self._preload_dataset(data_dir)

    def _preload_dataset(self, data_dir):
        # List all files in the specified directory
        file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Load and process all images
        for idx, file_path in enumerate(file_paths):
            image = read_image(file_path).float() / 255.0  # Normalize to [0, 1]
            self.images.append(image)
            self.labels.append(self.default_label)  # Use default_label for all

            # Store the image data as numpy for variance calculation
            np_image = image.permute(1, 2, 0).numpy()  # Convert CxHxW to HxWxC for numpy
            self.data.append(np_image)

        # Convert list of numpy arrays to a single numpy array
        self.data = np.array(self.data)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.preload:
            image = self.images[idx]
        else:
            raise RuntimeError("Dataset must be preloaded.")
        
        if self.transform:
            # Apply the transform if it's provided
            image = self.transform(image)

        label = self.labels[idx]
        return image.numpy(), label  # Ensure image is a NumPy array to be compatible with both PyTorch and Jax.




