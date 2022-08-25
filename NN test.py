import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.randn(64,64)

class train_model(nn.Module):
    def __init__(self): # 相当于定义self自身的_init_文档
        super(train_model,self).__init__()  # 定义网络，继承自身，引入由nn.module类定义的网络层
        self.linear_0 = nn.Linear(64,64)
        self.sigmoid = nn.Sigmoid()
        self.linear_1 = nn.Linear(64,16)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(16,1)
        self.linear_3 = nn.Linear(64,1)

    def forward(self,x):
        x_0 = torch.clone(x)
        x = self.linear_0(x)    # 定义前馈传播，调用已经引入的网络层
        x = self.sigmoid(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x_1 = self.linear_3(x_0)
        goal_out = torch.add(x,x_1)
        return goal_out


net_model = train_model()
net_model.eval()

torch.onnx.export(
    net_model,
    x,
    'net_work test.onnx',
    input_names=['x'],
    output_names=['goal_out'],
    dynamic_axes={'x':{0:'sample_batch'},'goal_out':{0:'goal_batch'}}
)

print(net_model)
