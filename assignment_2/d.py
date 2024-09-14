import torch
import torchvision
import matplotlib.pyplot as plt

mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float() / 255.0
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

mnist_train = torchvision.datasets.MNIST(root='./data', train=False, download=True)
x_test = mnist_train.data.reshape(-1, 784).float() / 255.0
y_test = torch.zeros((mnist_train.targets.shape[0], 10))
y_test[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

class MNIST_Model(torch.nn.Module):
    def __init__(self):
        super(MNIST_Model, self).__init__()
        self.linear = torch.nn.Linear(784, 10)
    
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)
    
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.forward(x), y)
    
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.forward(x).argmax(1), y.argmax(1)).float())

model = MNIST_Model()
optimizer = torch.optim.SGD(model.parameters(), lr=2.5)

for epoch in range(1000):
    loss = model.loss(x_train, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {model.loss(x_train, y_train).item():.4f}')

print('Loss:', model.loss(x_train, y_train).item())
print('Accuracy:', model.accuracy(x_test, y_test).item())

weights = model.linear.weight.data.numpy()

for i in range(10):
    plt.imshow(weights[i].reshape(28, 28), cmap='gray')
    plt.title(f'{i}')
    plt.savefig(f'./assignment_2/images/weights_{i}.png')