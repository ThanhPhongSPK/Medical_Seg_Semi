import numpy as np 
import os 
from scipy.ndimage import zoom, rotate
import h5py
import itertools

import torch 
from torch.utils.data import Dataset, Sampler

class BaseDataset(Dataset): 
    def __init__(self, root_path, split= 'train', transform= None, num= None): 
        """
        Use to load name of file.list 

        """
        self.root_path = root_path
        self.split = split 
        self.transform = transform
        self.sample_list = []

        if split == 'train': 
            file_name = os.path.join(self.root_path, 'train_slices.list')
            with open(file_name, 'r') as file: 
                self.sample_list = file.readlines() 
            
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
            # print(self.sample_list)
        elif split == 'val': 
            file_name = os.path.join(self.root_path, 'val.list')
            with open(file_name, 'r') as file: 
                self.sample_list = file.readlines() 
            
            self.sample_list = [item.replace('\n','') for item in self.sample_list]
            # print(self.sample_list)
        
        else: 
            raise ValueError(f'Mode: {self.split} is not supported')
        

        if isinstance(num, int): 
            self.sample_list = self.sample_list[:num]
        
        print(f'Total slices: {len(self.sample_list)}')
        

    def __len__(self): 
        return len(self.sample_list)

    def __getitem__(self, index):
        case = self.sample_list[index]

        # open the file.h5 
        if self.split == 'train': 
            file_path = os.path.join(self.root_path, f'data/slices/{case}.h5')
        elif self.split == 'val': 
            file_path = os.path.join(self.root_path, f'data/{case}.h5')
        
        h5f = h5py.File(file_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        sample = {'image': image, 'label': label}
        if self.split == 'train' and self.transform is not None: 
            sample = self.transform(sample)
        
        sample['case'] = case 
        return sample 
    
def random_rot_flip(image, label): 
    """
    Random rotate and Random flip 
    """
    
    # Random rotate
    k = np.random.randint(0, 4) 
    image = np.rot90(image, k)
    label = np.rot90(label, k)

    # Random flip 
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis).copy() 
    label = np.flip(label, axis).copy() 

    return image, label 

def random_rotate(image, label):
    angle = np.random.randint(-20, 20) 
    image = rotate(image, angle, order= 0, reshape= False)
    label = rotate(label,angle, order=0, reshape= False )
    return image, label

class RandomGenerator: 
    def __init__(self, output_size): 
        self.output_size = output_size
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.random() > 0.5: 
            image, label = random_rot_flip(image, label)
        
        if np.random.random() > 0.5: 
            image, label = random_rotate(image, label) 
        
        # Zoom image to -> [256,256]
        x,y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order= 0)
        label = zoom(label, (self.output_size[0] /x , self.output_size[1] / y), order= 0)

        # Convert to pytorch 
        imageTensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0) # image.shape = (1, H, W)
        labelTensor = torch.from_numpy(label.astype(np.uint8)) # label.shape = (H, W)
        sample = {'image': imageTensor, 'label': labelTensor}
        
        return sample
    
def iterate_once(indices): 
    """
    Permutate the iterable once 
    (permutate the labeled_idxs once)
    """
    return np.random.permutation(indices) 

def iterate_externally(indices): 
    """
    Create an infinite iterator that repeatedly permutes the indices.
    ( permutate the unlabeled_idxs to make different)
    """
    def infinite_shuffles(): 
        while True: 
            yield np.random.permutation(indices)
            
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n): 
    args = [iter(iterable)] * n 
    return zip(*args)

class TwoStreamBatchSampler(Sampler): 
    def __init__(self, primary_indicies, secondary_indicies, batchsize, secondary_batchsize): 
        self.primary_indicies = primary_indicies
        self.secondary_indicies = secondary_indicies
        self.primary_batchsize = batchsize - secondary_batchsize
        self.secondary_batchsize = secondary_batchsize

        assert len(self.primary_indicies) >= self.primary_batchsize > 0 
        assert len(self.secondary_indicies) >= self.secondary_batchsize > 0 

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indicies)
        secondary_iter = iterate_externally(self.secondary_indicies)

        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) 
            in zip(grouper(primary_iter, self.primary_batchsize),
                   grouper(secondary_iter, self.secondary_batchsize))
        )

    def __len__(self): 
        return len(self.primary_indicies) // self.primary_batchsize