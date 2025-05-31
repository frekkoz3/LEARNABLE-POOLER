"""
    This material is develop for academic purpose. 
    It is develop by Francesco Bredariol as final project of the Introduction to ML course (year 2024-2025).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedPooling2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, in_channels = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        # Gating mechanism: a sigmoid on the output of a conv2d
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute average and max pooling
        avg_pooled = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        max_pooled = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)

        # Compute gate values from input
        gate = self.sigmoid(x)  # shape: (B, C, H, W)
        gate = F.avg_pool2d(gate, kernel_size=self.kernel_size, stride=self.stride)

        # Blend max and average pooling using the gate
        return gate * max_pooled + (1 - gate) * avg_pooled

class MixingPooling2d(nn.Module):
    def __init__(self, in_channels = 1, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        # Learning mixing coefficient per channel
        self.alpha = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x):
        # Compute average and max pooling
        avg_pooled = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        max_pooled = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)

        # Sigmoid to let the alpha between in [0, 1]
        alpha = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        # Blend max and average pooling using the gate
        return alpha * max_pooled + (1 - alpha) * avg_pooled
    
if __name__ == "__main__":
    m = torch.ones((2, 2))
    print(F.sigmoid(m)*2)