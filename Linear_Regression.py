import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# define train_dataset and test_dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
# convert numpy.array into tensor
X_train = torch.from_numpy(x_train)
Y_train = torch.from_numpy(y_train)


# define Linear_Regression model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, input):
        x = self.linear(input)
        return x


# instanslize model
model = Model()
# define loss_function
loss_func = nn.MSELoss()
# define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# the process of trainning
epoches = 1000
for epoch in range(epoches):
    y_pred = model(X_train)
    loss = loss_func(y_pred, Y_train)

    # loss backward and update weight
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f'epoch: {epoch + 1}, loss: {loss:.8f}')

model.eval()
with torch.no_grad():
    y_pred = model(X_train)
preds = y_pred.data.numpy()

# plot the result
fig = plt.figure(figsize=(10, 5))
plt.plot(X_train.numpy(), Y_train.numpy(), 'ro', label='Original data')
plt.plot(X_train.numpy(), preds, label='Fitting Line')
# show the plot
plt.legend()
plt.show()

