import numpy as np 
import math
from itertools import combinations_with_replacement


#Mean Squared Error 

def mean_squared_error(predict, actual):
    """
    Computes mean squared error using predictions and ground truth
    """
    return np.sum((actual - predict)**2) / len(actual)


#Polynomial Expansion 

def polynomial_expansion(X, degree):
    '''
    Expands data matrix given chosen polynomial degree
    Polynomial Expansion function code adapted from eriklindernoren's ML-From-Scratch module 
    '''    
    n_samples, n_features = X.shape

    #Creates all possible combinations of feature cols using itertools 
    poly_combs = [] 
    for deg in range(degree + 1): 
        inter_combs = [combinations_with_replacement(range(n_features), deg)]
        poly_combs.extend(inter_combs)
            
    #convert to list and omit columns of ones for now - will reinsert later
    poly_combs = [item for sublist in poly_combs for item in sublist][1:]

    n_output_features = len(poly_combs)
    X_new = np.zeros((n_samples, n_output_features))

    #multiply all combinations of features by each other to generate poly expansion 
    for feat, comb in enumerate(poly_combs):  
        X_new[:, feat] = np.prod(X[:, comb], axis=1)

    return X_new


#Splitting and Cross Validation Functions 

def train_test_split(X, y, train_ratio=0.8):
    """
    Splits feature matrix and labels into training and testing sets
    """
    #Find number of samples and create random indices 
    num_samples = X.shape[0]
    train_samples = int(num_samples * train_ratio)
    rand_indices = np.random.permutation(num_samples) 

    #creates train and test arrays of indices based on subset size 
    train_idx, test_idx = rand_indices[:train_samples], rand_indices[train_samples:]

    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx, :], y[test_idx, :]

    return X_train, X_test, y_train, y_test 

def k_fold_splits(X, y, k): 
    """
    Splits the data into k training / validation permutations
    """

    num_samples = len(y)
    subset_size = int(num_samples / k) 
    start_pos = 0 
    end_pos = subset_size 

    folds = [] 

    for _ in range(k): 

        #creates subset of labels and feature matrix to hold out 
        subset_y = y[start_pos:end_pos]
        subset_X = X[start_pos:end_pos]
        
        #creates new training label and feature matrix without the hold out subset
        train_y = np.concatenate((y[:start_pos, :], y[end_pos:, :]), axis=0)
        train_X = np.concatenate((X[:start_pos, :], X[end_pos:, :]), axis=0)

        folds.append([train_X, train_y, subset_X, subset_y])
        start_pos += subset_size
        end_pos += subset_size

    return folds 