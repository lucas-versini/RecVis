from torchvision import datasets
from typing import Optional, Callable, Tuple, Any
import torch
from data.imagenet_r import ImageFolderSafe
import os
import numpy as np

class severalNoises:
    def __init__(self, root, batch_size: int = 1, steps_per_example: int = 1,
                 shuffle = False, order = None, transform: Optional[Callable] = None,
                 single_crop: bool = False):        
        self.root = root
        list_roots = os.listdir(root)
        list_roots = [os.path.join(root, x) for x in list_roots]
        print(f"Found {len(list_roots)} datasets")

        self.batch_size = batch_size
        self.steps_per_example = steps_per_example
        self.shuffle = shuffle
        self.transform = transform
        self.single_crop = single_crop

        self.list_datasets = []
        for root in list_roots:
            dataset = ExtendedImageFolder(root=root, transform=transform, single_crop=single_crop, batch_size=batch_size, steps_per_example=steps_per_example)
            self.list_datasets.append(dataset)
        
        self.len = sum([len(dataset) for dataset in self.list_datasets])
        self.cum_len = [0]
        for dataset in self.list_datasets:
            self.cum_len.append(self.cum_len[-1] + len(dataset) // batch_size)
        
        self.shuffle = shuffle
        if shuffle:
            if order is not None:
                self.order = order
            else:
                num_unique_images = self.len // (batch_size * steps_per_example)
                self.order_ = np.random.permutation(num_unique_images)
                self.order = self.order_ * steps_per_example
                self.order = np.concatenate([np.arange(n, n + steps_per_example) for n in self.order])

    def __len__(self):
        return self.len
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.shuffle:
            index = self.order[index]
        for i, cum_len in enumerate(self.cum_len):
            if index < cum_len:
                real_index = index - self.cum_len[i - 1]
                # print(i, cum_len, real_index)
                return self.list_datasets[i - 1][real_index]
        raise ValueError("Index out of bounds")

class ExtendedImageFolder(ImageFolderSafe):
    def __init__(self, root: str, batch_size: int = 1, steps_per_example: int = 1, minimizer = None, transform: Optional[Callable] = None, single_crop: bool = False, start_index: int = 0):
        super().__init__(root=root, transform=transform)
        self.batch_size = batch_size
        self.minimizer = minimizer
        self.steps_per_example = steps_per_example
        self.single_crop = single_crop
        self.start_index = start_index
    
    def __len__(self):
        mult = self.steps_per_example * self.batch_size
        mult *= (super().__len__() if self.minimizer is None else len(self.minimizer)) 
        return mult

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # print(index, self.steps_per_example, self.batch_size, super().__len__(), self.minimizer, len(self.samples))
        real_index = (index // self.steps_per_example) + self.start_index
        if self.minimizer is not None:
            real_index = self.minimizer[real_index]
        path, target = self.samples[real_index]
        sample = self.loader(path)
        if self.transform is not None and not self.single_crop:
            samples = torch.stack([self.transform(sample) for i in range(self.batch_size)], axis=0)
        elif self.transform and self.single_crop:
            s = self.transform(sample)
            samples = torch.stack([s for i in range(self.batch_size)], axis=0)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return samples, target


class ExtendedSplitImageFolder(ExtendedImageFolder):
    def __init__(self, root: str, batch_size: int = 1, steps_per_example: int = 1, split: int = 0, minimizer = None, 
                 transform: Optional[Callable] = None, single_crop: bool = False, start_index: int = 0):
        super().__init__(root=root, batch_size=batch_size, steps_per_example=steps_per_example, minimizer=minimizer, 
                         transform=transform, single_crop=single_crop, start_index=start_index)
        self.new_samples = []
        for i, sample in enumerate(self.samples):
            if i % 20 == split:
                self.new_samples.append(sample)
        self.samples = self.new_samples