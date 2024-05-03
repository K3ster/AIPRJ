import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

# Load and preprocess the data
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

# Define the neural network model
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

# Convert the data to PyTorch tensors
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

# Train the model with different hyperparameters
lr_vec = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7 ])
K1_vec = np.arange(1,11,2)
K2_vec = K1_vec
PK_2D_K1K2 = np.zeros([len(K1_vec),len(K2_vec)])
max_epoch = 200
PK_2D_K1K2_max = 0
k1_ind_max = 0
k2_ind_max = 0

for k1_ind in range(len(K1_vec)):
    for k2_ind in range(len(K2_vec)):
        model = Model(X_train.shape[1], int(max(y)+1), K1_vec[k1_ind], K2_vec[k2_ind])
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
            PK_2D_K1K2[k1_ind, k2_ind] = PK

        if PK > PK_2D_K1K2_max:
            PK_2D_K1K2_max = PK
            k1_ind_max = k1_ind
            k2_ind_max = k2_ind

# Fit the KMeans model and generate cluster labels for the training data only
from sklearn.decomposition import PCA

# Apply PCA to the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train.numpy())

# Fit the KMeans model and generate cluster labels for the PCA-transformed data
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_pca)

# Get the cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Create a scatter plot of the PCA-transformed data with different colors for different clusters
for i in range(2):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Klaster {i+1}')

# Plot the cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='red', label='Centra klastrów')

# Show the plot with a legend
plt.legend()
plt.show()

# Algorytm k-średnich (k-Means) to popularna metoda grupowania danych. Działa on na podstawie następujących kroków:

# Inicjalizacja: Wybieramy k punktów danych losowo jako początkowe centra klastrów.
# Przypisanie do klastrów: Dla każdego punktu danych, przypisujemy go do najbliższego centrum klastra. Najczęściej używana miara odległości to odległość euklidesowa.
# Aktualizacja centrów klastrów: Dla każdego klastra, obliczamy średnią wszystkich punktów danych należących do tego klastra i ustawiamy to jako nowe centrum klastra.
# Powtarzanie kroków 2 i 3: Powtarzamy kroki 2 i 3 do momentu, gdy centra klastrów przestaną się zmieniać lub po określonej liczbie iteracji.
# Na koniec algorytmu, mamy k klastrów z punktami danych przypisanymi do klastrów na podstawie ich podobieństwa do siebie nawzajem.


#Czy można użyć domyślnych modeli żeby pokazać różnicę