import torch
import torch.nn as nn
from sine_layer import SineLayer

class VoxelMLP_Sine(nn.Module):
    """
    SIREN Voxel MLP Model
    """
    def __init__(self, config):
        super(VoxelMLP_Sine, self).__init__()
        self.config = config
        
        layers = []
        
        # --- 1. 输入层 -> 第一个隐藏层 (SIREN) ---
        # 注意: SIREN 第一层通常需要较大的 omega_0 (比如 30) 来引入高频分量
        layers.append(SineLayer(
            config.input_dim, 
            config.hidden_dims[0], 
            is_first=True, 
            omega_0=30.0  # 第一层频率系数
        ))
        
        # --- 2. 隐藏层循环 (SIREN) ---
        for i in range(1, len(config.hidden_dims)):
            layers.append(SineLayer(
                config.hidden_dims[i-1], 
                config.hidden_dims[i], 
                is_first=False, 
                omega_0=1.0  # 后续层通常设为 1.0
            ))
            
        # --- 3. 最后一层 -> 输出层 (Linear) ---
        # 最后一层通常不需要激活函数，或者是 Sigmoid (如果你想把颜色限制在 0-1)
        # 这里保持线性，用于输出 RGB
        final_linear = nn.Linear(config.hidden_dims[-1], config.output_dim)
        
        # 手动初始化最后一层 (因为它不是 SineLayer，需要 Xavier 初始化)
        with torch.no_grad():
            nn.init.xavier_uniform_(final_linear.weight)
            nn.init.constant_(final_linear.bias, 0.0)
            
        layers.append(final_linear)
        
        # 打包模型
        self.model = nn.Sequential(*layers)
        
        # 注意：这里删除了 self._init_weights() 调用
        # 因为 SineLayer 已经在内部初始化好了，再次 Xavier 会破坏它。

    def forward(self, x):
        # return self.model(x)
        output = self.model(x)
        return torch.sigmoid(output)  # 确保输出在 [0, 1] 范围内