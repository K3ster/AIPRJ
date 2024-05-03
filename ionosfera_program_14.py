import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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

depth_vec = np.arange(1,111,2)
PK_1D_depth = np.zeros(len(depth_vec))

for depth_ind in range(len(depth_vec)):
    model = DecisionTreeClassifier(max_depth=depth_vec[depth_ind])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    PK = accuracy_score(y_test, y_pred)*100
    print("Depth {} | PK {}". format(depth_vec[depth_ind], PK))
    PK_1D_depth[depth_ind] = PK

plt.figure(figsize=(8, 6))
plt.plot(depth_vec, PK_1D_depth, marker='o')
plt.xlabel('Depth')
plt.ylabel('PK')
plt.title('PK vs Depth for Decision Trees')
plt.grid(True)
plt.savefig("Fig.1_PK_Depth_decisiontree_ionosphere_with_tree.png",bbox_inches='tight')
#92.95-wynik
