import os

from torch.utils.data import DataLoader
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler

class DistributedDataParellelUtils():
    def setup(self):

        # initialize the process group
        dist.init_process_group(backend="nccl")

    def cleanup(self):
        dist.destroy_process_group()
    
    def prepare(self, train_set, batch_size, pin_memory=False, num_workers=0):

        '''
        Description: Divide the whole dataset among all GPUs. Dataset // world_size
        '''
        sampler = DistributedSampler(train_set, num_replicas=int(os.environ['WORLD_SIZE']),  
                        shuffle=False, drop_last=False)
                        # Rank of the current process within num_replicas.(optional)
                        # drop_last : to divide among GPUs, sampler tries to do equal partition, thus in case when 
                        # size of dataset % world_size != 0 , sampler either adds 0 (drop_last = false, default)
                        # or drops the last index (drop_last) = True 
                        
        
        dataloader = DataLoader(train_set, batch_size=batch_size, pin_memory=pin_memory,
                     num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
        return dataloader