import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x_train = torch.tensor([[0.0], [1.0]])
y_train = torch.tensor([[1.0], [0.0]])

x_test = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]])
y_test = torch.tensor([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])

class NOT_Model(nn.Module):
    def __init__(self):
        super(NOT_Model, self).__init__()
        self.W = nn.Parameter(torch.tensor([[0.0]]))
        self.b = nn.Parameter(torch.tensor([[0.0]]))
    
    def logits(self, x):
        return x @ self.W + self.b
    
    def forward(self, x):
        return torch.sigmoid(self.logits(x))
    
    def loss(self, x, y):
        return nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)
    
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.forward(x).argmax(1), y.argmax(1)).float())
    
model = NOT_Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10000):
    loss = model.loss(x_train, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/10000], Loss: {model.loss(x_train, y_train).item():.4f}')

print('W = %s, b = %s, loss = %s' % (model.W.item(), model.b.item(), model.loss(x_train, y_train).item()))
print('Accuracy: %s' % model.accuracy(x_test, y_test).item())

plt.scatter(x_train, y_train, label='$(x^{(i)},y^{(i)})$')
x_plot = torch.arange(0, 1.02, 0.02).reshape(-1, 1)
plt.plot(x_plot, model.forward(x_plot).detach(), label='$\\hat y = f(x) = \\sigma(xW+b)$', color='red')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()