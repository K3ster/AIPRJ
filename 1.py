import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, K1, K2, dropout_p=0.5):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, K1),
            nn.BatchNorm1d(K1),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(K1, K2),
            nn.BatchNorm1d(K2),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.layer3 = nn.Linear(K2, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.softmax(self.layer3(x), dim=1)
        return x

lr_vec = np.arange(0, 1.01, 0.05)
K1_vec = np.arange(1,111,15)
K2_vec = np.arange(1,111,15)
PK_2D_K1K2 = np.zeros([len(K1_vec),len(K2_vec), len(lr_vec)])
max_epoch = 200

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

avg_PK = np.zeros(len(lr_vec))  # Dodano tablicę do przechowywania średnich PK

for lr_ind in range(len(lr_vec)):
    for k1_ind in range(len(K1_vec)):
        for k2_ind in range(len(K2_vec)):
            model = Model(X_train.shape[1], int(max(y)+1), K1_vec[k1_ind], K2_vec[k2_ind])
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_vec[lr_ind])
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
                PK_2D_K1K2[k1_ind, k2_ind, lr_ind] = PK

    avg_PK[lr_ind] = np.mean(PK_2D_K1K2[:, :, lr_ind])  # Obliczanie średniego PK dla danego learning rate

# Tworzenie wykresu zależności średniego PK od learning rate
plt.figure(figsize=(8, 6))
plt.plot(lr_vec, avg_PK, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Średnie PK')
plt.title('Zależność średniego PK od Learning Rate')
plt.grid(True)
plt.savefig("Fig.2_PK_vs_LR_pytorch_ionosphere.png",bbox_inches='tight')
