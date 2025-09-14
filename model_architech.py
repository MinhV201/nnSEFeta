import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional
#from torchinfo import summary

class feta_encoder(nn.Module):
    def __init__(self, inputs: int, filter: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=inputs, out_channels=filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter),
            nn.ReLU(),
            nn.Conv2d(in_channels=filter, out_channels=filter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        x = self.block(x)
        return x
    
# Tạo lớp kiến trúc thêm các đặc trưng còn lại 
class Residual(nn.Module):
    def __init__(self, block: nn.Module = nn.Sequential, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut 

    def forward(self, x: Tensor) ->Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x


class InvertedResidual(nn.Sequential):
    def __init__(self, inputs: int, outputs: int, stride: int, expansion: int = 6):
        expand_features = expansion*inputs
        super().__init__(
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        #Block_res_1
                        nn.Conv2d(in_channels=inputs, out_channels=expand_features, kernel_size=1, padding=0),
                        nn.GroupNorm(4, expand_features),
                        nn.ReLU6(inplace=True),
                        #Block_res_2                           
                        nn.Conv2d(in_channels=expand_features, out_channels=expand_features, kernel_size=3, stride=stride, padding=1),
                        nn.GroupNorm(4, expand_features),
                        nn.ReLU6(inplace=True),
                        #Block_res_3
                        nn.Conv2d(in_channels=expand_features, out_channels=outputs, kernel_size=1, padding=0),
                        nn.GroupNorm(4, outputs),
                        nn.Identity()
                    ),
                    shortcut= nn.Sequential(
                        nn.Conv2d(in_channels=inputs, out_channels=outputs, kernel_size=1, stride=stride, padding=0),
                        nn.GroupNorm(4, outputs),
                        #nn.ReLU6()
                    )
                    if inputs != outputs or stride != 1
                    else None,
                ),
                nn.ReLU6(inplace=True),
            )
        )

class SEBlock(nn.Module):
    def __init__(self, input_channels, internal = None):
        if internal is None : internal = max(1, input_channels // 8) 
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, (1, 1)) #global average pooling2d
        x = self.down(x)
        x = F.relu(x, inplace=True) # nn.SiLu (Sigmoid Linear Unit)
        x = self.up(x)
        x = torch.sigmoid(x)
        #x = x.view(-1, self.input_channels, 1, 1)
        return x * inputs
    
class FETA(nn.Module):
    def __init__(self, input_channels: int, out_channels: int):
        super(FETA, self).__init__()
        self.stage1 = nn.Sequential(
            feta_encoder(input_channels, 32), # 224x224 -> 112x112
            feta_encoder(32, 64)#112x112 -> 56x56
        )
        self.stage2 = nn.Sequential(
            #feta_encoder(64, 128), # 14x14
            InvertedResidual(64, 128, stride=2, expansion=2), #28x28
            #SEBlock(128),
            InvertedResidual(128, 128, stride=1, expansion=4), 
            SEBlock(128)
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(128, 128, stride=2, expansion=6), # 14x14
            #SEBlock(128),
            InvertedResidual(128, 256, stride=1, expansion=6),
            SEBlock(256)
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout = nn.Dropout(0.4)
        self.fc_class = nn.Linear(256, out_channels)
        self.fc_qualify = nn.Linear(256, out_channels)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = self.Dropout(x)
        x = torch.flatten(x, 1)
        out_class = self.fc_class(x)
        out_qualify = self.fc_qualify(x)
        return out_class, out_qualify
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FETA(input_channels=3, out_channels=3)
    model = model.to(device)
    summary(model, input_size = (1, 3, 224, 224))
    #dummy = torch.randn(1, 1, 224, 224)
    #result = model(dummy)
    #print(result)