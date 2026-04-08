
import torch
import torch.nn as nn

class PhysicalPowerCurve(nn.Module):

    def __init__(self, device):
        super(PhysicalPowerCurve, self).__init__()
        self.device = device
        

        DATA_MAX_SPEED = 14.12
        self.v_in = 3.0 / DATA_MAX_SPEED
        self.v_rated = 12.5 / DATA_MAX_SPEED

        self.v_out = 25.0 / DATA_MAX_SPEED 

        self.p_rated = 1.0

        self.k = self.p_rated / (self.v_rated ** 3 + 1e-8)

    def forward(self, wind_speed):

        v = torch.clamp(wind_speed, min=0.0)
        

        mask_zero = (v < self.v_in) | (v > self.v_out)

        mask_rated = (v >= self.v_rated) & (v <= self.v_out)

        mask_ramp = ~(mask_zero | mask_rated)
        
        # 3. 计算输出
        p_phys = torch.zeros_like(v)
        
        # 应用爬坡公式
        if mask_ramp.any():
            p_phys[mask_ramp] = self.k * (v[mask_ramp] ** 3)
            
        # 应用满发公式
        if mask_rated.any():
            p_phys[mask_rated] = self.p_rated
            
        # mask_zero 区域默认为 0，不需要额外操作
        
        return p_phys