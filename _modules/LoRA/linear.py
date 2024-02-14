import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, alpha, lora_dropout=0.):
        super().__init__()
        
        self.linear = linear_layer
        
        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features
        self.rank = rank
        self.alpha = alpha
        
        std_dev = 1 / torch.sqrt(torch.tensor(self.rank).float())
        
        self.A = nn.Parameter(torch.randn(self.in_features, self.rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(self.rank, self.out_features))
        self.dropout = nn.Dropout(lora_dropout)
        
    def forward(self, x):
        
        x1 = self.linear(x)
        x2 = self.alpha * (x @ self.A @ self.B)
        x2 = self.dropout(x2)
        return x1 + x2