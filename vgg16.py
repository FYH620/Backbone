from torch import nn
from torch.nn import functional as F

class VGG16(nn.Module):
    def __init__(self,num_classes):
        super(VGG16,self).__init__()
        self.features=self._make_conv()
        self.classifier=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
        )
    
    def _make_conv(self):
        layers=[]
        in_dims=3
        out_dims=64
        for i in range(1,19):
            if i in [3,6,10,14,18]:
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
                if i!=14 and i!=18:
                    out_dims=out_dims*2
            else:
                layers.append(nn.Conv2d(in_dims,out_dims,kernel_size=3,stride=1,padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_dims=out_dims
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        x=F.softmax(x,dim=1)
        return x

#vgg16=VGG16(num_classes=1000)