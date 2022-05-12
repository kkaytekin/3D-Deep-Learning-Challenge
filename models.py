import torch
import torch.nn as nn
import torch.nn.functional as F


class point_model(nn.Module):
    ### Complete for task 1
    def __init__(self,num_classes):
        super(point_model, self).__init__()
        self.modules = nn.ModuleList()
        self.modules.append(nn.Conv1d(3,64,1))
        self.modules.append(nn.Conv1d(64,128,1))
        self.modules.append(nn.Conv1d(128,512,1))
        self.modules.append(nn.Linear(512,64))
        self.modules.append(nn.Linear(64,8))
    def forward(self,x):
        for module in self.modules:
            x = module(x)
        return x
        

class voxel_model(nn.Module):
    ### Complete for task 2
    def __init__(self,num_classes):
        super(voxel_model, self).__init__()
        self.conv1=nn.Conv3d(1,8,3)

    def forward(self,x):

        return x
        

class spectral_model(nn.Module):
    ### Complete for task 3
    def __init__(self,num_classes):
        super(spectral_model, self).__init__()
        self.mlp1=nn.Conv1d(6,64,1)

    def forward(self,x):

        return x
