import torch
from torch import nn

class InceptionV1(nn.Module):
    def __init__(self,in_dims,out_1_1_dims,out_2_1_dims,out_2_dims,out_3_1_dims,out_3_2_dims,out_4_2_dims):
        super(InceptionV1,self).__init__()
        self.branch1x1=nn.Sequential(
            nn.Conv2d(in_dims,out_1_1_dims,kernel_size=1,stride=1),
            nn.ReLU()
        )
        self.branch3x3=nn.Sequential(
            nn.Conv2d(in_dims,out_2_1_dims,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(out_2_1_dims,out_2_dims,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.branch5x5=nn.Sequential(
            nn.Conv2d(in_dims,out_3_1_dims,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(out_3_1_dims,out_3_2_dims,kernel_size=5,stride=1,padding=2),
            nn.ReLU()
        )
        self.branch_pool=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_dims,out_4_2_dims,kernel_size=1,stride=1),
            nn.ReLU()
        )

    def forward(self,x):
        x1=self.branch1x1(x)
        x2=self.branch3x3(x)
        x3=self.branch5x5(x)
        x4=self.branch_pool(x)
        output=torch.cat([x1,x2,x3,x4],dim=1)
        return output

