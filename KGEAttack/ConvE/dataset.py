'''
1. Dataset structure -- s,r,one-hot-labels, mode
2. Dataset class takes in (s,r) pairs,their labels and mode and returns one-hot encoded vectors for labels and s,r as two separate vectors
'''

from torch.utils.data import Dataset
from typing import Dict, Tuple, List
import numpy as np
import torch

class TrainDataset(Dataset):
    def __init__(self, args, num_ent, sr2o:Dict[Tuple[int, int], List[int]], mode:str):
        '''
        Input can be sr2o or or2s
        Mode is 'lhs' for or2s and 'rhs' for sr2o
        '''
        self.sr2o = sr2o
        self.sr = list(self.sr2o.keys())
        self.args = args
        self.n_ent = num_ent
        self.entities = np.arange(self.n_ent, dtype=np.int32)
        self.mode = mode 
        #mode is not needed for generating data, but needed in data iterator to decide direction for model.forward()
        
    def __len__(self):
        return len(self.sr)
    
    def __getitem__(self, idx):
        sample_key = self.sr[idx]
        s,r = int(sample_key[0]), int(sample_key[1])
        index_target = np.array(self.sr2o[(s,r)], dtype=np.int32)
        sample_label = self.get_label(index_target)
        s,r = torch.tensor(sample_key[0], dtype=torch.long), torch.tensor(sample_key[1], dtype=torch.long)
        index_target = torch.tensor(index_target, dtype=torch.long)
        # label smoothing
        if self.args.label_smoothing != 0.0:
            sample_label = (1.0 - self.args.label_smoothing)*sample_label + (1.0/self.n_ent)
            
        return s,r,sample_label, self.mode
        
        
    @staticmethod
    def collate_fn(data):
        s = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        #index_target = torch.stack([_[2] for _ in data], dim=0) #this gives error
        label = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]

        return s, r,label, mode
    
    def get_label(self, index_target:List[int]):
        # get the multi-one-hot labels from indices
        one_hot = np.zeros(self.n_ent, dtype=np.float32)
        np.add.at(one_hot, index_target, 1.0)
        return torch.FloatTensor(one_hot)
    
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_lhs, dataloader_rhs):
        #self.iterator_lhs = iter(dataloader_lhs)
        #self.iterator_rhs = iter(dataloader_rhs)
        self.iterator_lhs = self.one_shot_iterator(dataloader_lhs)
        self.iterator_rhs = self.one_shot_iterator(dataloader_rhs)
        self.step = 0
        
    def __next__(self):
        if self.step % 2 == 0:
            data = next(self.iterator_lhs)
        else:
            data = next(self.iterator_rhs)
        
        self.step += 1
        return data
    
    def __iter__(self): 
        return self
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
           
        
    