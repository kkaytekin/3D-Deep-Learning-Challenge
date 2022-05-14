import torch
import torch.nn as nn
import torch.nn.functional as F


class point_model(nn.Module):
    ### Complete for task 1
    def __init__(self,num_classes):
        super(point_model, self).__init__()
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 =  nn.Linear(1024,512)
        self.fc2 =  nn.Linear(512,256)
        self.fc3 =  nn.Linear(256,num_classes)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    def forward(self,x):
        x = x.transpose(1,2)
        xb = F.relu(self.bn1(self.conv1(x)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))
        xb = self.fc3(xb)
        return xb
        
## Architecture 1
# class voxel_model(nn.Module):
#     ### Complete for task 2
#     def __init__(self,num_classes):
#         super(voxel_model, self).__init__()
#         self.conv1=nn.Conv3d(1,8,3)
#         self.conv2=nn.Conv3d(8,32,3)
#         self.conv3=nn.Conv3d(32,128,3)
#         self.fc1=nn.Linear(128,128)
#         self.fc2=nn.Linear(128,num_classes)
#
#         self.bn1 = nn.BatchNorm3d(8)
#         self.bn2 = nn.BatchNorm3d(32)
#         self.bn3 = nn.BatchNorm3d(128)
#         self.bn4 = nn.BatchNorm1d(128)
#
#     def forward(self,x):
#         x = x.unsqueeze(1)
#         xb = F.relu(self.bn1(self.conv1(x)))
#         # xb = F.avg_pool3d(xb,2)
#         xb = F.relu(self.bn2(self.conv2(xb)))
#         # xb = F.avg_pool3d(xb,2)
#         xb = F.relu(self.bn3(self.conv3(xb)))
#         xb = F.max_pool3d(xb,xb.size(-1))
#         xb = nn.Flatten()(xb)
#         xb = F.relu(self.bn4(self.fc1(xb)))
#         xb = self.fc2(xb)
#         return xb

# Architecture 2
class voxel_model(nn.Module):
    ### Complete for task 2
    def __init__(self,num_classes):
        super(voxel_model, self).__init__()
        self.conv1=nn.Conv3d(1,8,6,stride=2)
        self.conv2=nn.Conv3d(8,32,5,stride=2)
        self.conv3=nn.Conv3d(32,128,4)
        self.fc1=nn.Linear(1024,128)
        self.fc2=nn.Linear(128,num_classes)

        self.bn1 = nn.BatchNorm3d(8)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm1d(128)

        self.do1 = nn.Dropout3d(p=0)
        self.do2 = nn.Dropout3d(p=0)
        self.do3 = nn.Dropout(p=0)

    def forward(self,x):
        x = x.unsqueeze(1)
        xb = F.leaky_relu(self.bn1(self.conv1(x)))
        xb = F.leaky_relu(self.do1(self.bn2(self.conv2(xb))))
        xb = F.leaky_relu(self.do2(self.bn3(self.conv3(xb))))
        # xb = F.max_pool3d(xb,2)
        xb = nn.Flatten()(xb)
        xb = F.leaky_relu(self.do3(self.bn4(self.fc1(xb))))
        xb = self.fc2(xb)
        return xb

# Architecture 3: 3D Shape Net. less than 20% acc.
# class voxel_model(nn.Module):
#     ### Complete for task 2
#     def __init__(self,num_classes):
#         super(voxel_model, self).__init__()
#         self.conv1=nn.Conv3d(1,48,6,stride=2)
#         self.conv2=nn.Conv3d(48,160,5,stride=2)
#         self.conv3=nn.Conv3d(160,512,4)
#         self.fc1=nn.Linear(512,128)
#         self.fc2=nn.Linear(128,num_classes)
#
#         self.bn1 = nn.BatchNorm3d(48)
#         self.bn2 = nn.BatchNorm3d(160)
#         self.bn3 = nn.BatchNorm3d(512)
#         self.bn4 = nn.BatchNorm1d(128)
#
#     def forward(self,x):
#         x = x.unsqueeze(1)
#         xb = F.relu(self.bn1(self.conv1(x)))
#         xb = F.relu(self.bn2(self.conv2(xb)))
#         xb = F.relu(self.bn3(self.conv3(xb)))
#         xb = F.max_pool3d(xb,2)
#         xb = nn.Flatten()(xb)
#         xb = F.relu(self.bn4(self.fc1(xb)))
#         xb = self.fc2(xb)
#         return xb

class spectral_model(nn.Module):
    ### Complete for task 3
    def __init__(self,num_classes):
        super(spectral_model, self).__init__()
        self.conv1=nn.Conv1d(6,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 =  nn.Linear(1024,512)
        self.fc2 =  nn.Linear(512,256)
        self.fc3 =  nn.Linear(256,num_classes)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    def forward(self,x):
        x = x.transpose(1,2)
        xb = F.relu(self.bn1(self.conv1(x)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten()(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))
        xb = self.fc3(xb)
        return xb
