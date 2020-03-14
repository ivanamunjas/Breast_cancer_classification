# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:14:13 2019

@author: Ivana Pesic & Ivana Munjas
"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt

def neuralNetwork(X,Y):
    
    # =============================================================================
    # KREIRANJE TRAINING I TEST SKUPA
    # =============================================================================

    benign = []
    malignant = []
    
    for i in range(0,699):
        if Y[i] == 2 :
            benign.append(np.asarray([X[i,:]]))
        else:
            malignant.append(np.asarray([X[i,:]]))
    
    indB = np.random.permutation(len(benign))
    numB = round(0.9*len(indB))
    
    indM = np.random.permutation(len(malignant))
    numM = round(0.9*len(indM))
    
    benign_train = [benign[::][x] for x in indB[0:numB]]
    benign_test = [benign[::][x] for x in indB[numB::]]
    
    malignant_train = [malignant[::][x] for x in indM[0:numM]]
    malignant_test = [malignant[::][x] for x in indM[numM::]]
    
    Xtrain = benign_train + malignant_train
    Xtest = benign_test + malignant_test
    
    ytrain = [0]*len(benign_train) + [1]*len(malignant_train)
    ytest = [0]*len(benign_test) + [1]*len(malignant_test)
    
    indTrain = np.random.permutation(len(Xtrain))
    indTest = np.random.permutation(len(Xtest))
    
    Xtrain = [Xtrain [::][x] for x in indTrain]
    Xtest = [Xtest [::][x] for x in indTest]
    ytrain = [ytrain [::][x] for x in indTrain]
    ytest = [ytest [::][x] for x in indTest]
    
    x_train = np.asarray(Xtrain[::][0:round(0.8*len(Xtrain))])
    x_val = np.asarray(Xtrain[::][round(0.8*len(Xtrain)):len(Xtrain)])
    y_train = np.asarray(ytrain[::][0:round(0.8*len(ytrain))])
    y_val = np.asarray(ytrain[::][round(0.8*len(ytrain)):len(ytrain)])
    x_test = np.asarray(Xtest)
    y_test = np.asarray(ytest)
    
    layer_sizes = [32, 64, 128]
    activation_functions = [tf.nn.relu, tf.nn.softmax]
    class_weights = [{0: 1, 1: 2}, {0: 1, 1: 1}, {0: 0.5, 1: 1}]
    #class_weights = [{0: 1, 1: 2}]
    best_acc = 0
    best_layer_size = 0
    best_ac_fnc = tf.nn.relu
    best_class_weight = {0: 1, 1: 2}
    best_f1 = 0
    
    # =============================================================================
    # KROSVALIDACIJA
    # =============================================================================
    
    for layer_size in layer_sizes:
        for act_function in activation_functions:
            for class_weight in class_weights:
                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.Flatten())
                model.add(tf.keras.layers.Dense(layer_size, activation = act_function))
                model.add(tf.keras.layers.Dense(layer_size, activation = act_function))
                model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
                
                model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
                model.fit(x_train, y_train, epochs = 3, class_weight = class_weight)
                
                val_loss, val_acc = model.evaluate(x_val, y_val)
                y_val_pred = model.predict(x_val)
                y_val_pred_1D = []
                for i in y_val_pred:
                    if i[0] > i[1]:
                        y_val_pred_1D.append(0)
                    else:
                        y_val_pred_1D.append(1)
                
                matrix = confusion_matrix(y_val, y_val_pred_1D)
                
                precision = matrix[0][0]/(matrix[0][1] + matrix[0][0])
                recall = matrix[0][0]/(matrix[1][0] + matrix[0][0])
                
                if np.isnan(precision) or np.isnan(recall):
                    f1= 0
                else:
                    f1 = 2*precision*recall/(precision + recall)     
                
                
                acc = (matrix[0][0] + matrix[1][1])/sum(sum(matrix))
                
                if (best_f1 < f1) and (not np.isnan(f1)):
                    best_f1 = f1
                    best_acc = acc
                    best_layer_size = layer_size
                    best_ac_fnc = act_function
                    best_class_weight = class_weight
                    
    
    print('----------------------------------------------' + '\n' + '*** Best model ***' + '\n' + \
         'Number of layers: ' + str(best_layer_size) + '\n' + 'Activation function: ' + str(best_ac_fnc) + \
           '\n' + 'Best wights of classes are: ' + str(best_class_weight) + '\n' + 'Accuracy:' + str(best_acc))
    
    # =============================================================================
    # TRENIRANJE S NAJBOLJIM PARAMETRIMA
    # =============================================================================
    
    x_train = np.asarray(Xtrain)
    y_train = np.asarray(ytrain)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(best_layer_size, activation = best_ac_fnc))
    model.add(tf.keras.layers.Dense(best_layer_size, activation = best_ac_fnc))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train, y_train, epochs = 3, class_weight = best_class_weight)
    y_train_pred_1D = []
    y_train_pred = model.predict(x_train)
    for i in y_train_pred:
        if i[0] > i[1]:
            y_train_pred_1D.append(0)
        else:
            y_train_pred_1D.append(1)
    
    matrix = confusion_matrix(y_train, y_train_pred_1D)
    plt.figure()
    ax = plt.subplot()
    seaborn.heatmap(matrix, annot=True, ax = ax); 
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix - train'); 
    ax.xaxis.set_ticklabels(['malign', 'benign']); ax.yaxis.set_ticklabels(['malign', 'benign']);
    
                
    precisionTrain = matrix[0][0]/(matrix[0][1] + matrix[0][0])
    recallTrain = matrix[0][0]/(matrix[1][0] + matrix[0][0])
    accTrain = (matrix[0][0] + matrix[1][1])/sum(sum(matrix))
    
    # =============================================================================
    # TESTIRANJE MREZE
    # =============================================================================
    
    y_test_pred = model.predict(x_test)
    y_test_pred_1D = []
    for i in y_test_pred:
        if i[0] > i[1]:
            y_test_pred_1D.append(0)
        else:
            y_test_pred_1D.append(1)
    
    matrix = confusion_matrix(y_test, y_test_pred_1D)
    
    plt.figure()
    ax = plt.subplot()
    seaborn.heatmap(matrix, annot=True, ax = ax); 
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix - test'); 
    ax.xaxis.set_ticklabels(['benign', 'malign']); ax.yaxis.set_ticklabels(['benign', 'malign']);
    
    precision = matrix[0][0]/(matrix[0][1] + matrix[0][0])
    recall = matrix[0][0]/(matrix[1][0] + matrix[0][0])
    
    f1 = 2*precision*recall/(precision + recall)
    acc = (matrix[0][0] + matrix[1][1])/sum(sum(matrix))
    
    print('-----------------------------------------------' + '\n' + 'Results on the test set:' + \
          '\n' + 'Accuracy:' + str(acc))
    
    return acc, precision, recall, matrix, precisionTrain, recallTrain, accTrain