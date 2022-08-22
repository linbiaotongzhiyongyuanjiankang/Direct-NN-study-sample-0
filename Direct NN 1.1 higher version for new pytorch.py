import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(5)
'''This version of GAN test file NN 1.1 is prepared for pytorch >= 1.4.0: It should be like calculating all leaf_joints 
before any optimizer.step() works, for example as we are involving with a problem with a trigger proceed through 
generator 1 (G1) and then generator 2 (G2) , we shall calculate the final loss value , and then use backward engine from
 pytorch or any kind of our own engines to get the grad_modified tensors of G1 and G2, after all we can use 
 optimizer.step() to modify G1 , then G2 . '''
x1 = 64
x2 = 32
x3 = x1
x4 = x1

hid1 = 64

dorate = 0.0001

ptspos = np.vstack([np.linspace(-1, 1, x2) for _ in range(x1)])  # (x1, x2)

a = np.random.uniform(1, 2, size=x1)[:, np.newaxis]

paints = a * np.power(ptspos, 2) + a - 1

paints = torch.from_numpy(paints).float()

G1 = nn.Sequential(
    nn.Linear(x2, hid1),
    nn.ReLU(),
    nn.Linear(hid1, x2)
)

G2 = nn.Sequential(
    nn.Linear(x2, x2),
    nn.ReLU()
)

D = nn.Sequential(
    nn.Linear(x2, x2),
    nn.Linear(x2, 1),
    nn.Sigmoid()
)

optimizer_G1 = torch.optim.Adam(G1.parameters(), lr=dorate)
optimizer_G2 = torch.optim.Adam(G2.parameters(), lr=dorate)
optimizer_D = torch.optim.Adam(D.parameters(), lr=dorate)

plt.ion()

res1 = []
res2 = []
res3 = []

for epoch in range(15000):

    trigger = torch.randn(x1, x2)
    Goal1 = G1(trigger)
    Goal2 = G2(Goal1)
    G3 = torch.add(Goal1, Goal2)

    D_pro = D(paints)
    DG3 = D(G3)

    G_3loss = torch.mean(torch.log(1. - DG3))
    Gsanloss = -torch.reciprocal(G_3loss)
    Dloss = -torch.mean(torch.log(D_pro) + torch.log(1. - DG3))
    optimizer_D.zero_grad()
    Dloss.backward(retain_graph=True)


    G_3loss = torch.mean(torch.log(1. - DG3))
    Gsanloss = -torch.reciprocal(G_3loss)
    optimizer_G2.zero_grad()
    Gsanloss.backward(retain_graph=True)


    G_3loss = torch.mean(torch.log(1. - DG3))
    Gsanloss = -torch.reciprocal(G_3loss)
    optimizer_G1.zero_grad()
    Gsanloss.backward()

    optimizer_D.step()
    optimizer_G2.step()
    optimizer_G1.step()




    print("Eopch:{}, Loss:{:.3f}".format(epoch + 1, Dloss))

    if epoch % 1000 == 0:
        res1 = res1 + [epoch]
        k = Dloss.detach().numpy()
        p = Gsanloss.detach().numpy()
        res2 = res2 + [k]
        res3 = res3 + [p]

        plt.clf()
        plt.plot(res1, res2, c='#4AD631', lw=3, label='Dloss painting', )
        plt.plot(res1, res3, c='#583AFF', lw=3, label='Gloss painting', )
        plt.text(-.5, 2.3, 'D accuracy=%.3f (0.5 for D to converge)' % D_pro.detach().numpy().mean(),
                 fontdict={'size': 13})
        plt.ylim((0, 5))
        plt.legend(loc='upper right', fontsize=10)  # 标签图例
        plt.draw()
        plt.pause(0.1)

plt.ioff()
plt.show()

'''
G1  ...(n,n)...>G2 \
                    ...(+)...>G3 
    ...mul(k)...>  /

'''
