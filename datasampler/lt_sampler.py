import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random



"""======================================================"""
REQUIRES_STORAGE = False

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, opt, image_dict, image_list, **kwargs):
        self.pars = opt

        #####
        self.image_dict         = image_dict
        self.image_list         = image_list

        #####
        self.classes        = list(self.image_dict.keys())

        ####
        self.batch_size         = opt.bs
        self.samples_per_class  = opt.samples_per_class
        self.sampler_length     = len(image_list)//opt.bs
        assert self.batch_size%self.samples_per_class==0, '#Samples per class must divide batchsize!'
        
        
        self.class_count = {key:len(self.image_dict[key]) for key in self.image_dict.keys()}
        self.class_full_lst = [key for key, count in self.class_count.items() for _ in range(count)]                  
        
        self.name             = 'lt_sampler'
        self.requires_storage = False

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from Random classes
            draws = self.batch_size//self.samples_per_class

            for _ in range(draws):               
                class_key = random.choice(self.class_full_lst)
                class_ix_list = [random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)]             
                # class_ix_list = [idx1, idx2] idx는 전체 dataset에 대해서 정해짐
                subset.extend(class_ix_list)

            yield subset

    def __len__(self):
        return self.sampler_length
