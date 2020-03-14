# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:41:58 2019

@author: Ivana Munjas
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from NN import neuralNetwork
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('C:\\Users\\ASUS\\Desktop\\ETF\\IV\\I semestar\\Neuralne mreze\\Projekat NM\\wcdb.csv')

# =============================================================================
# REDUKCIJA DIMENZIJA 
# =============================================================================

X = data.values[:,1:10]
X1 = X
X = StandardScaler().fit_transform(X)
Y = data.values[:,10]
mean_vec = np.mean(X, axis=0)
cov_mat = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0]-1)
cov_mat = np.cov(X.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# pravljenje liste parova sopstvenih vektora i vrednosti
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# sortiranje po opadajucem redosledu
eig_pairs.sort(key=lambda x: x[0], reverse=True)
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

# koliko informacija nosi koja komponenta
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 9))
    plt.bar(range(9), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

matrix_w = np.hstack((eig_pairs[0][1].reshape(9,1),
                      eig_pairs[1][1].reshape(9,1),
                      eig_pairs[2][1].reshape(9,1),
                      eig_pairs[3][1].reshape(9,1),
                      eig_pairs[4][1].reshape(9,1),
                      eig_pairs[5][1].reshape(9,1),
                      eig_pairs[6][1].reshape(9,1),
                      eig_pairs[7][1].reshape(9,1)))


X2 = X.dot(matrix_w[:,0:4])
X3 = X.dot(matrix_w)

# parametri za razlicite vrednosti dimenzija ulaznih podataka: 9, 4 i 8
acc1, prec1, recall1, matrix1, precT1, recallT1, accT1 = neuralNetwork(X1,Y)
acc2, prec2, recall2, matrix2, precT2, recallT2, accT2  = neuralNetwork(X2,Y)
acc3, prec3, recall3, matrix3, precT3, recallT3, accT3  = neuralNetwork(X3,Y)