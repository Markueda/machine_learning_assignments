import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = torch.tensor([[0.0], [1.0], [1.0], [1.0]])

class NAND_Model(nn.Module):
    def __init__(self):
        super(NAND_Model, self).__init__()
        self.W = nn.Parameter(torch.tensor([[0.0], [0.0]]))
        self.b = nn.Parameter(torch.tensor([[0.0]]))
    
    def forward(self, x):
        return torch.sigmoid(x @ self.W + self.b)
    
    def loss(self, x, y):
        return nn.functional.mse_loss(self.forward(x), y)
    
model = NAND_Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10000):
    loss = model.loss(x_train, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/10000], Loss: {loss.item():.4f}')

wItems = model.W.detach().numpy()
print('W = %s, b = %s, loss = %s' % (wItems, model.b.item(), model.loss(x_train, y_train).item()))

detail = 10
x_1 = torch.linspace(0, 1, detail)
x_2 = torch.linspace(0, 1, detail)
x_1, x_2 = torch.meshgrid(x_1, x_2, indexing='ij')
x = torch.cat((x_1.reshape(-1, 1), x_2.reshape(-1, 1)), dim=1)

with torch.no_grad():
    y = model(x).reshape(detail, detail)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

surface = ax.plot_surface(x_1.numpy(), x_2.numpy(), y.numpy(), cmap='viridis')
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
ax.scatter(points[:, 0].numpy(), points[:, 1].numpy(), points[:, 2].numpy(), color='red', label='$(x^{(i)},y^{(i)})$')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.legend()
plt.show()