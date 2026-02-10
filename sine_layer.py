import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """
    SIREN 的核心层：Linear + Sine 激活 + 特殊初始化
    Paper: Implicit Neural Representations with Periodic Activation Functions
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=20):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # 第一层初始化范围: (-1/in_features, 1/in_features)
                lim = 1 / self.linear.weight.size(1)
                self.linear.weight.uniform_(-lim, lim)
            else:
                # 后续层初始化范围: (-sqrt(6/in_features)/omega_0, sqrt(6/in_features)/omega_0)
                lim = np.sqrt(6 / self.linear.weight.size(1)) / self.omega_0
                self.linear.weight.uniform_(-lim, lim)
            
            # 偏置初始化为 0
            if self.linear.bias is not None:
                self.linear.bias.zero_()
                
    def forward(self, input):
        # 公式: sin(omega_0 * (Wx + b))
        return torch.sin(self.omega_0 * self.linear(input))