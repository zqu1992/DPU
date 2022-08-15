import torch
import numpy as np   
import time


class DPU_module(object):
    """Implement the DPU module for conv/fc layers.

    Args:
        w_init (tensor): the initial weight tensor, also the start point of the first step
        layer_idx (int): the layer index
        
    """
    def __init__(self, w_init, layer_idx):
        # Record the tensor shape
        self.tensor_shape = w_init.size()
        # Copy the values in the initial weight tensor
        self.w_init = w_init.detach().clone().to('cuda')
        # The layer index
        self.layer_idx = layer_idx
        # Initialize the binary mask m as one, i.e., all weights are trainable
        self.mask = torch.ones_like(self.w_init).to(torch.bool)
        # Initialize the rewinding metric as zero
        self.metric = torch.zeros_like(self.w_init)
        # Initialize the local contribution as zero
        self.local_contribution = torch.zeros_like(self.w_init)
        # Initialize the global contribution as zero
        self.global_contribution = torch.zeros_like(self.w_init)
        # The number of updated weights in this layer
        self.num_updated_weights = torch.sum(self.mask).item()

    def sort_metric(self, num_top):
        """Sort the metric (only the valid metric with mask value 1) and output a list that contains the information of top num_top weights"""
        valid_metric = (self.metric[self.mask]).view(-1)
        sorted_idx = torch.argsort(valid_metric, descending=True)
        if num_top > self.num_updated_weights:
            top_list = torch.tensor([ [ self.layer_idx, sorted_idx[i], valid_metric[sorted_idx[i]] ] for i in range(self.num_updated_weights)])  
        else:
            top_list = torch.tensor([ [ self.layer_idx, sorted_idx[i], valid_metric[sorted_idx[i]] ] for i in range(num_top)])  
        return top_list

    def update_mask(self, act_list):
        """Update the mask according to the given list of active weights"""
        mask_new = self.mask.clone()
        valid_idx = torch.nonzero(mask_new.view(-1))
        mask_new.view(-1)[valid_idx[act_list]]=0
        self.mask.logical_xor_(mask_new)
        self.num_updated_weights = torch.sum(self.mask).item()        
        return True 

    def replace_mask(self, mask_new):
        """Replace the mask with the mask_new"""
        self.mask.copy_(mask_new)   
        self.num_updated_weights = torch.sum(self.mask).item()    
        return True 