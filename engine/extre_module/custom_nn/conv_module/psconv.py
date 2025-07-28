import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   
   
import warnings 
warnings.filterwarnings('ignore')    
from calflops import calculate_flops     
   
import torch    
import torch.nn as nn    
import torch.nn.functional as F
import numpy as np
from engine.extre_module.ultralytics_nn.conv import Conv, autopad
 
class PSConv(nn.Module):   
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):   
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]   
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0) 

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))  
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))     
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
