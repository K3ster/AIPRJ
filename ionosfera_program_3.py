import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

x,y_t,x_norm = hkl.load('ionosphere.hkl')
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
    def __init__(self, input_dim, output_dim, K1, K2):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, K1)
        self.layer2 = nn.Linear(K1, K2)
        self.layer3 = nn.Linear(K2, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

lr_vec = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7 ])
K1_vec = np.arange(1,101,1)
PK_vec = np.zeros(len(K1_vec))
max_epoch = 200
K2_vec = K1_vec
PK_2D_K1K2 = np.zeros([len(K1_vec),len(K2_vec)])

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

for k1_ind in range(len(K1_vec)):
    for k2_ind in range(len(K2_vec)):
        model = Model(X_train.shape[1], int(max(y)+1), K1_vec[k1_ind], K2_vec[k1_ind])
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
        print("K1 {} | K2 {} | PK {}". format(K1_vec[k1_ind], K2_vec[k2_ind], PK))
        PK_vec[k1_ind] = PK

# plt.figure(figsize=(8, 6))
# plt.plot(K1_vec, PK_vec, 'o-')
# plt.xlabel('K1')
# plt.ylabel('PK')
# plt.title('Zmiana poprawności podczas zmiany neuronów w pierwszej warstwie')
# plt.grid(True)
# plt.savefig("Fig.1_PK_K1_pytorch_ionosphere.png",bbox_inches='tight')

# Normalizacja wartości PK


import pylab
from sklearn.cluster import KMeans

# Assuming X_train is a 2D tensor
X_train_np = X_train.detach().numpy()

# Use KMeans to cluster the data
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train_np)
y_kmeans = kmeans.predict(X_train_np)

# Plot the data and the clustering effect
pylab.scatter(X_train_np[:, 0], X_train_np[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot the centroids
centers = kmeans.cluster_centers_
pylab.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

pylab.show()



# Normalizacja wartości PK
PK_vec_norm = (PK_vec - np.min(PK_vec)) / (np.max(PK_vec) - np.min(PK_vec))

# Wykres liniowy
plt.figure(figsize=(8, 6))
plt.plot(K1_vec, PK_vec_norm, 'o-')
plt.xlabel('K1')
plt.ylabel('PK (normalized)')
plt.title('Normalized PK vs K1')
plt.grid(True)
plt.savefig("Fig.1_PK_K1_pytorch_ionosphere_norm.png",bbox_inches='tight')

# Histogram
plt.figure(figsize=(8, 6))
sns.histplot(PK_vec_norm, kde=True, color='red')  # Dodanie histogramu
plt.xlabel('PK (normalized)')
plt.title('Distribution of PK')
plt.grid(True)
plt.savefig("Fig.2_Distribution_PK_pytorch_ionosphere.png",bbox_inches='tight')

