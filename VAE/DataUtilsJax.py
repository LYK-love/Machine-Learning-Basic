import numpy as np
import os
from imageio import imread


class ImageDatasetForJAX:
    def __init__(self, root_dir, train=True, default_label=0, preload=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            train (bool): True = training set, False = test set.
            default_label (int): Label for all images since no labels are provided.
            preload (bool): If True, preload the images into memory.
        """
        self.root_dir = os.path.join(root_dir, 'train' if train else 'test')
        self.default_label = default_label
        self.images = []
        self.labels = []
        self.preload = preload

        if self.preload:
            self._preload()

    def _preload(self):
        for fname in os.listdir(self.root_dir):
            path = os.path.join(self.root_dir, fname)
            if path.endswith(('.png', '.jpg', '.jpeg')):
                # Read and normalize image
                image = imread(path).astype(np.float32) / 255.0
                image = (image - 0.5) / 1.0  # Normalization
                self.images.append(image)
                self.labels.append(self.default_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if not self.preload:
            raise RuntimeError("Dataset must be preloaded.")
        return self.images[idx], self.labels[idx]
