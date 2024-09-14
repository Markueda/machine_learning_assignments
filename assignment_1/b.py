import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('day_length_weight.csv')

x_train = torch.tensor(df[['length']].values, dtype=torch.float32).reshape(-1, 1)
z_train = torch.tensor(df[['weight']].values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(df[['day']].values, dtype=torch.float32).reshape(-1, 1)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.W = nn.Parameter(torch.tensor([[0.0], [0.0]]))
        self.b = nn.Parameter(torch.tensor([[-5.0]]))
    
    def forward(self, x):
        return x @ self.W + self.b
    
    def loss(self, x, y):
        return nn.functional.mse_loss(self.forward(x), y)
    
x_combined = torch.cat((x_train, y_train), dim=1)

model = LinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001)

for epoch in range(1000):
    loss = model.loss(x_combined, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

w_items = model.W.detach().numpy()
print('W = %s, b = %s, loss = %s' % (w_items, model.b.item(), model.loss(x_combined, y_train).item()))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(x_train, y_train, z_train, 'o', markersize=1, color='lightblue')
ax.plot_surface(x_train, model.forward(x_combined).detach(), z_train, color='red', label='Fitted plane')

ax.set_xlabel('Day')
ax.set_ylabel('Length')
ax.set_zlabel('Weight')
plt.legend()
plt.show()


