import torch
import numpy as np

epoch_max = 1024
dorate1 = 0.0000125
dorate2 = 0.000025
data_batch = 64
data_spec = 4

s = torch.randn(data_batch,data_spec)

hid1 = 8
w1 = torch.randn(data_spec,hid1)

hid2 = 8
w2 = torch.randn(hid1,hid2)

hid3 = 8
w3 = torch.randn(hid2,hid3)

hid4 = 3
w4 = torch.randn(hid3,hid4)

data_output = hid4
y = torch.randn(data_batch,data_output)

for epoch in range(0,epoch_max):

    h1 = s.mm(w1)
    h2 = h1.mm(w2)
    h2 = h2.clamp(min=0)
    h3 = h2.mm(w3)
    h3 = h3.clamp(min=0)
    Y = h3.mm(w4)

    Y_loss = (Y-y).pow(2).sum()
    Y_loss = np.around(Y_loss,decimals=3)
    print("Eopch:{}, Y_Loss:{:.3f}".format(epoch + 1, Y_loss))
    plt.ioff()
    plt.show()

    grad_Y = 2*(Y-y)

    # Y = h3 * w4
    grad_w4 = h3.t().mm(grad_Y)
    grad_h3 = grad_Y.mm(w4.t())

    # h3 = h2 * w3
    grad_w3 = h2.t().mm(grad_h3)
    grad_h2 = grad_h3.mm(w3.t())

    # h2 = h1 * w2
    grad_w2 = h1.t().mm(grad_h2)
    grad_h1 = grad_h2.mm(w2.t())

    # h1 = s * w1
    grad_w1 = s.t().mm(grad_h1)

    w1 = w1 - dorate1 * grad_w1
    w2 = w2 - dorate1 * grad_w2
    w3 = w3 - dorate2 * grad_w3
    w4 = w4 - dorate2 * grad_w4

print("w1",w1)
print("w2",w2)
print("w3",w3)
print("w4",w4)

'''
    神经网络正向输出的简图：
    s---(w1)--->h1---(w2)--->h2---(w3)--->h3---(w4)--->Y
    
    神经网络反向传播梯度的简图：
     Y--->(w4)--->h3--->(w3)--->h2--->(w2)--->h1--->(w1)
     
    隐层w1、w2共用 dorate1
    隐层w3、w4共用 dorate2
'''
