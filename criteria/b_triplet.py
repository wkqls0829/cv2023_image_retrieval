import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False

### Standard Triplet Loss, finds triplets in Mini-batches.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.margin     = opt.loss_triplet_margin
        self.batchminer = batchminer
        self.name           = 'b_triplet'
        self.img_num_per_cls = opt.img_num_per_cls
        self.img_num_total = np.sum(self.img_num_per_cls)

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def triplet_distance(self, anchor, positive, negative):
        return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def forward(self, batch, labels, **kwargs):
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        sampled_triplets = self.batchminer(batch, labels)
        loss             = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) * self.img_num_total / self.img_num_per_cls[labels[triplet[0]]] for triplet in sampled_triplets])

        return torch.mean(loss)
