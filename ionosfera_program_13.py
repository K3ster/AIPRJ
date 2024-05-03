import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

K_vec = np.arange(1,111,2)
PK_1D_K = np.zeros(len(K_vec))

for k_ind in range(len(K_vec)):
    model = KNeighborsClassifier(n_neighbors=K_vec[k_ind])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    PK = accuracy_score(y_test, y_pred)*100
    print("K {} | PK {}". format(K_vec[k_ind], PK))
    PK_1D_K[k_ind] = PK

plt.figure(figsize=(8, 6))
plt.plot(K_vec, PK_1D_K, marker='o')
plt.xlabel('K')
plt.ylabel('PK')
plt.title('PK vs K for K-Nearest Neighbors')
plt.grid(True)
plt.savefig("Fig.1_PK_K_knn_ionosphere_Neihgbor.png",bbox_inches='tight')
