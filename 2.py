import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

x,y_t,x_norm = hkl.load('ionosphere2.hkl')
if min(y_t.T)[0] > 0:
    y=y_t.squeeze()-1
else:
    y=y_t.squeeze()
X=x.T

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, layers, dropout_p=0.5):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, layers[i]))
            else:
                self.layers.append(nn.Linear(layers[i-1], layers[i]))
            self.layers.append(nn.BatchNorm1d(layers[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_p))
        self.layers.append(nn.Linear(layers[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x, dim=1)

lr_vec = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7 ])
max_epoch = 200

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

PK_values = []
layer_counts = list(range(11, 31))

for layer_count in layer_counts:
    layers = [100]*layer_count
    model = Model(X_train.shape[1], int(max(y)+1), layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_vec[0])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(max_epoch):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = model(X_test)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        PK = correct.mean().item()*100
        PK_values.append(PK)

plt.plot(layer_counts, PK_values)
plt.xlabel('Liczba warstw')
plt.ylabel('Średnie PK')
plt.title('Wykres średniego PK w zależności od liczby warstw')
plt.savefig("Fig.1_PK_Layers_pytorch_ionosphere2.png",bbox_inches='tight')
plt.show()
