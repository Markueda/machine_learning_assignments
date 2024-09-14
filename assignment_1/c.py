import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('day_head_circumference.csv')

x_train = torch.tensor(df[['day']].values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(df[['head circumference']].values, dtype=torch.float32).reshape(-1, 1)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.W = nn.Parameter(torch.tensor([[0.0]]))
        self.b = nn.Parameter(torch.tensor([[0.0]]))

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def forward(self, x):
        return 20 * self.sigmoid(x @ self.W + self.b) + 31
    
    def loss(self, x, y):
        return nn.functional.mse_loss(self.forward(x), y)

model = LinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001)

for epoch in range(10000):
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/10000], Loss: {loss.item():.4f}')

print('W = %s, b = %s, loss = %s' % (model.W.item(), model.b.item(), model.loss(x_train, y_train).item()))

plt.scatter(x_train, y_train, s=1, label='Data', color='lightblue')

x_min, x_max = torch.min(x_train), torch.max(x_train)
x_range = torch.linspace(x_min, x_max, 100).reshape(-1, 1)
y_range = model.forward(x_range).detach()
plt.plot(x_range.numpy(), y_range.numpy(), label='Fitted line', color='red')

plt.xlabel('Day')
plt.ylabel('Head Circumference')
plt.legend()
plt.show()


