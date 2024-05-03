# Listing1.py
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
filename = 'ionosphere.txt'
data = np.loadtxt(filename, delimiter=',', dtype=str)
x = data[:,0:-1].astype(float).T
y_t = data[:,-1].astype(float)
y_t = y_t.reshape(1,y_t.shape[0]) # dla kodowanie klas naturalno-liczbowego
# min i max dla każdej z cech przed normalizacją
print(np.transpose([np.array(range(1,x.shape[0]+1)), x.min(axis=1), x.max(axis=1)]))
# normalizacja do przedziału <-1,1> wg zależnoci
# x_norm = (x_norm_max-x_norm_min)*(x-x_min)/(x_max-x_min) + x_norm_min
# w której x_norm_max oraz x_norm_min są docelowymi wartociami rozpiętoci cechy
x_min = x.min(axis=1)
x_max = x.max(axis=1)
x_norm_max = 1
x_norm_min = -1
x_norm = np.zeros(x.shape)
for i in range(x.shape[0]):
    x_norm[i,:] = (x_norm_max-x_norm_min)/(x_max[i]-x_min[i])* \
     (x[i,:]-x_min[i]) + x_norm_min
# sprawdzenie rozpiętosci cech po normalizacji
print(np.transpose([np.array(range(1,x.shape[0]+1)), x_norm.min(axis=1),
x_norm.max(axis=1)]))
plt.plot(y_t[0])
#hkl.dump([x,y_t,x_norm],'ionosphere.hkl')
#x,y_t,x_norm = hkl.load('ionosphere.hkl')
#x = x_norm # zakomentowanie tej linii pokaże wpływ normalizacji na zbieżnosć procesu
#print(data[:,-1].astype(float))
#print(x)
xcorr=pd.DataFrame(x)
xcorr=xcorr.corr()
print(xcorr)
xcorr.to_string()
np.savetxt('corr.txt',xcorr)
