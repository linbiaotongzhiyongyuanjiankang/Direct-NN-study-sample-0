import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.randn(64,16)

class train_model(nn.Module):
    def __init__(self): # 相当于定义self自身的_init_文档
        super(train_model,self).__init__()  # 定义网络，继承自身，引入由nn.module类定义的网络层
        self.linear_0 = nn.Linear(16,32)
        self.linear_1 = nn.Linear(32,16)
        self.linear_2 = nn.Linear(16,1)
        # linear sets
        self.linear_3 = nn.Linear(16,1)
        # linear sets
        self.judge_0 = nn.Linear(1,16)
        self.judge_1 = nn.Linear(16,1)
        # iudge sets
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # spec sets

    def forward(self,x):
        x_0 = torch.clone(x)
        x = self.linear_0(x)    # 定义前馈传播，调用已经引入的网络层
        x = self.sigmoid(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x_1 = self.linear_3(x_0)
        x_1 = self.relu(x_1)
        goal = torch.add(x,x_1)
        goal = self.judge_0(goal)
        goal = self.relu(goal)
        goal = self.judge_1(goal)
        goal_out = self.sigmoid(goal)
        return goal_out


net_model = train_model()
net_model.eval()

torch.onnx.export(
    net_model,
    x,
    'net_work test.onnx',
    export_params=False,
    input_names=['x'],
    output_names=['goal_out'],

)

print(net_model)

'''
In PyTorch, linear computing is defined as goal.t() = weight[infeature,output].t() * trigger[sample_batch,infeature].t() + bias[sample_batch,output].t()
That's why we would find linear weights are displaying as being transposed in Netron
'''

