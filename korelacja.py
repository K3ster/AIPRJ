import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import pandas as pd

x,y_t,x_norm = hkl.load('ionosphere.hkl')
if min(y_t.T)[0] > 0:
    y=y_t.squeeze()-1
else:
    y=y_t.squeeze()
X=x.T

# Obliczanie współczynnika korelacji Pearsona
correlation_coefficients = []
for i in range(X.shape[1]):
    corr, _ = pearsonr(X[:, i], y)
    correlation_coefficients.append(corr)

# Wybieranie atrybutów z dodatnim współczynnikiem korelacji
positive_corr_indices = [i for i, corr in enumerate(correlation_coefficients) if corr > 0]
X_positive_corr = X[:, positive_corr_indices]

# Zapisywanie zmiennych x i y do pliku hkl
hkl.dump([x, y_t, x_norm], 'ionosphere2.hkl')

# Zapisywanie zmiennych x i y do pliku CSV
df = pd.DataFrame(np.hstack((X_positive_corr, y.reshape(-1, 1))), columns=[f'x{i+1}' for i in range(X_positive_corr.shape[1])] + ['y'])
df.to_csv('data2.csv', index=False)
