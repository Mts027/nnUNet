import pytest
import torch
from nnunetv2.training.loss.instance_losses import binary_cc
from nnunetv2.utilities.connected_components import get_voronoi

def first_example_test():
    gt = torch.zeros((1, 2, 3, 3, 3)) 
    pred = torch.zeros((1, 2, 3, 3, 3)) 

    gt[0,0,0,0,0] = 1
    
    voronoi = get_voronoi(y=gt, do_bg=False)

    score = binary_cc(y_pred=pred, y=gt, voronoi=voronoi[:,0,...], metric=lambda p,d: 0, activation=None)

    assert score == 0
    
