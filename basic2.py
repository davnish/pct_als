import torch
import torch.nn as nn 
# from numpy import asarray
# from numpy import savetxt
import numpy as np
import time

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels // 4, kernel_size=1)
        # self.trans_conv = nn.Conv1d(channels, channels // 4, kernel_size=1)
        self.after_norm = nn.BatchNorm1d()
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # print(self.k_conv.weight.shape)
        # b, n, n
        energy = torch.bmm(x_q, x_k)
        # print(energy.shape)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        print(x_r.shape)
        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = self.act(self.after_norm(x_r))
        # x = x + x_r
        return x
    
if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.rand((8, 128, 4096))
    head1 = SA_Layer(128)
    y = head1(x)
    # print(y.shape)