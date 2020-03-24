from copy import deepcopy
from torch import nn
import torch


class CatScaleShift(nn.Module):

    def __init__(self, in_channels, init_weight=1.0, init_bias=0.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = nn.Parameter(torch.tensor([init_weight]*in_channels*2))
        self.bias = nn.Parameter(torch.tensor([init_bias]*in_channels*2))
        # self.register_parameter('scale', self.weight)
        # self.register_parameter('shift', self.bias)
        
    def forward(self, input):
        return torch.cat((input,-input), dim=1) * self.weight.view(1,-1,1,1) + self.bias.view(1,-1,1,1)


class CReLUSequence(nn.Module):
    
    def __init__(self, in_c, out, ks=3, stride=1):
        super().__init__()
        
        self.down_1x1 = conv(in_c, out[0], 1) #scale down 1x1 conv
        self.cat = conv(out[0], out[1], ks, stride=stride, concat=True) #cat negative conv
        self.up_1x1 = conv(out[1]*2, out[2], 1) #scale up 1x1 conv
        
        if in_c!=out[-1]: 
            self.skip_adaptive = nn.Conv2d(in_c, out[-1], 1, stride=stride)
        else:
            self.skip_adaptive = nn.Identity()
        
    def forward(self, input):
        #res = self.fw(input) 
        #return self.fw(input)  + self.skip_adaptive(input)
        return self.up_1x1(self.cat(self.down_1x1(input))) + self.skip_adaptive(input)
        
            
class InceptionBlock(nn.Module):
    
    def __init__(self, in_c, out, stride=1):
        super().__init__()
        self.paths = nn.ModuleList()
        self.paths.append(conv(in_c, 64, 1, stride))
        
        self.paths.append(nn.Sequential(
                        conv(in_c, out[0], 1, stride),
                        conv(out[0], out[1], 3) ))
        
        self.paths.append(nn.Sequential(
                        conv(in_c, out[2], 1, stride),
                        conv(out[2], out[3], 3),
                        conv(out[3], out[3], 3) ))
        
        no_channels = 64+out[1]+out[3]
        
        if stride==2: 
            self.paths.append(nn.Sequential(
                        nn.MaxPool2d(3, stride, padding=1),
                        conv(in_c, 128, 1) ))
            no_channels += 128
                                     
        self.cat_conv = conv(no_channels, out[4], 1) 
        
        if in_c!=out[-1]: 
            self.skip_adaptive = nn.Conv2d(in_c, out[-1], 1, stride=stride)
        else:
            self.skip_adaptive = nn.Identity()
            
    def forward(self, input):
        #pdb.set_trace()
        all_tensors = [path(input) for path in self.paths]
        cat_tensors = torch.cat(all_tensors, dim=1)
        return self.cat_conv(cat_tensors) + self.skip_adaptive(input)
        

def conv(in_c, out_c, ks, stride=1, concat=False, activation=nn.ReLU()):
    layers = nn.ModuleList([
              nn.Conv2d(in_c, out_c, ks, stride, ks//2), 
              nn.BatchNorm2d(out_c), 
              ])
    if concat: layers.add_module('Concatenation', CatScaleShift(out_c))
    layers.add_module('activation', activation)
    return nn.Sequential(*layers)


def pvanet():
    return nn.Sequential(
        conv(3, 16, ks=7, stride=2, concat=True), #out is 32 cause concat
        nn.MaxPool2d(3, stride=2, padding=1),

        deepcopy(CReLUSequence(32, [24, 24, 64])),
        CReLUSequence(64, [24, 24, 64]),
        CReLUSequence(64, [24, 24, 64]),
        
        CReLUSequence(64, [48, 48, 128], stride=2),
        CReLUSequence(128, [48, 48, 128]),
        CReLUSequence(128, [48, 48, 128]),
        CReLUSequence(128, [48, 48, 128]),
        
        InceptionBlock(128, [48, 128, 24, 48, 256], stride=2),
        InceptionBlock(256, [64, 128, 24, 48, 256]),
        InceptionBlock(256, [64, 128, 24, 48, 256]),
        InceptionBlock(256, [64, 128, 24, 48, 256]),
        
        InceptionBlock(256, [96, 192, 32, 64, 384], stride=2),
        InceptionBlock(384, [96, 192, 32, 64, 384]),
        InceptionBlock(384, [96, 192, 32, 64, 384]),
        InceptionBlock(384, [96, 192, 32, 64, 384]),
        )
