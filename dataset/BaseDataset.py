import numpy as np 
import os 
from scipy.ndimage import zoom, rotate
import h5py

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