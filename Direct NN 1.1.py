import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(5)

x1 = 64
x2 = 64

x3 = x1
x4 = x1

hid1 = 128

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
    nn.Linear(x2, 1),
    nn.Sigmoid()
)

optimizer_G1 = torch.optim.Adam(G1.parameters(), lr=dorate)
optimizer_G2 = torch.optim.Adam(G2.parameters(), lr=dorate)
optimizer_D = torch.optim.Adam(D.parameters(), lr=dorate)

plt.ion()

res1 = []
res2 = []

for epoch in range(10000):

    trigger = torch.randn(x1, x2)
    Goal1 = G1(trigger)
    Goal2 = G2(Goal1)
    G3 = torch.add(Goal1, Goal2)

    D_pro = D(paints)
    DG3 = D(G3)

    kGsanloss = torch.mean(torch.log(1. - DG3))
    Gsanloss = -torch.reciprocal(kGsanloss)

    Dloss = -torch.mean(torch.log(D_pro) + torch.log(1. - DG3))

    optimizer_G1.zero_grad()
    Gsanloss.backward(retain_graph=True)
    optimizer_G1.step()

    optimizer_G2.zero_grad()
    Gsanloss.backward(retain_graph=True)
    optimizer_G2.step()

    optimizer_D.zero_grad()
    Dloss.backward()
    optimizer_D.step()

    print("Eopch:{}, Loss:{:.3f}".format(epoch + 1, Dloss))

    if epoch % 100 == 0:
        res1 = res1 + [epoch]
        k = Dloss.detach().numpy()
        res2 = res2 + [k]

        plt.clf()
        plt.plot(res1, res2, c='#4AD631', lw=3, label='vdloss painting', )
        plt.text(-.5, 2.3, 'D accuracy=%.3f (0.5 for D to converge)' % D_pro.data.numpy().mean(), fontdict={'size': 13})
        # plt.text(-.5, 2, 'G_loss= %.2f ' % G_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 5))
        plt.legend(loc='upper right', fontsize=10)
        plt.draw()
        plt.pause(0.2)

plt.ioff()
plt.show()
