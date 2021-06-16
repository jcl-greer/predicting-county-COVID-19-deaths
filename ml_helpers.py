import numpy as np 
import math
from itertools import combinations_with_replacement
from Utils import mean_squared_error, polynomial_expansion, train_test_split, k_fold_splits

#MODELS 

def gradient_descent(X, y, stepsize=0.001, iterations=1000, threshold=0.01):
    """
    Performs gradient descent to find optimal w
    """
    #randomly initialize weights 
    n_features = X.shape[1]
    limit = 1 / math.sqrt(n_features)
    w_old = np.random.uniform(-limit, limit, (n_features, ))

    errors = [] 

    count = 0 
    #loop through n iterations
    for _ in range(iterations):
        y_pred = X@(w_old)
        mse = mean_squared_error(y_pred, y)
        errors.append(mse)

        w_new = w_old - 2*stepsize*X.T@(y_pred - y)

        #stopping condition 
        if np.sum(abs(w_new - w_old)) < threshold:
            print(count) 
            return w_new, w_old, errors 
        w_old = w_new 

        count += 1

    return w_new, w_old, errors 

def least_squares_svd(X, y):
    """
    Uses SVD to compute optimal weights
    """
    U, S, V = np.linalg.svd(X)

    S_mat = np.zeros((X.shape[0], X.shape[1]))
    S_mat[:len(S), :len(S)] = np.diag(S)
    pseudo_inv = np.linalg.pinv(S_mat)
    w_hat = ((V.T@pseudo_inv)@U.T)@y 

    return w_hat 


def ls_linear_reg(X_train, X_test, y_train, y_test, grad_descent=False):
    """
    Performs least squares regression 
    Optional parameters to perform cross validation / gradient descent
    """

    #add column of ones to feature mats for constant
    X_train = np.insert(X_train, 0, 1, axis=1)
    X_test = np.insert(X_test, 0, 1, axis=1)

    #perform gradient descent if necessary; otherwise use svd to find what 
    if grad_descent:
        w_hat = gradient_descent(X_train, y_train)
    else:
        w_hat = least_squares_svd(X_train, y_train)

    #Apply to testing data and check error 
    error = ls_linear_reg_predict(X_test, y_test, w_hat)

    return error, w_hat 


def ls_linear_reg_predict(X_test, y_test, w_hat):
    """
    Creates predictions using the test data and optimal weights and generates error
    """
    predict = X_test@w_hat
    error = mean_squared_error(predict, y_test)

    return error 

def ridge_regression(X_train, X_test, y_train, y_test, param):
    """
    Performs least squares regression with a regularizer 
    """ 
    X_train = np.insert(X_train, 0, 1, axis=1)
    X_test = np.insert(X_test, 0, 1, axis=1)
    U, S, V = np.linalg.svd(X_train)
    full_s = np.zeros((X_train.shape[0], X_train.shape[1]))
    full_s[:len(S), :len(S)] = np.diag(S)
    
    StS = full_s.T@full_s
    ident = np.identity(StS.shape[0])
    
    w_hat = V.T@np.linalg.inv(StS + param*ident)@full_s.T@U.T@y_train

    error = ridge_regression_predict(X_test, y_test, w_hat)

    return error, w_hat 


def ridge_regression_predict(X_test, y_test, w_hat):
    """
    Creates predicts for ridge regression using the test data and optimal weights
    """

    predict = X_test@w_hat
    error = mean_squared_error(predict, y_test)

    return error 


def polynomial_regression(X_train, X_test, y_train, y_test, degree=2):
    """
    Applies polynomial expansion and calls least squares regression function 
    """

    poly_x_train = polynomial_expansion(X_train, degree)
    poly_x_test = polynomial_expansion(X_test, degree)

    poly_error, poly_w = ls_linear_reg(poly_x_train, poly_x_test, y_train, y_test)

    return poly_error, poly_w


def polynomial_ridge(X_train, X_test, y_train, y_test, degree=2, param=0.1):
    """
    Applies Polynomial expansion and calls ridge regression function 
    """

    poly_x_train = polynomial_features(X_train, degree)
    poly_x_test = polynomial_features(X_test, degree)

    poly_error, poly_w = ridge_regression(poly_x_train, poly_x_test, y_train, y_test, param)

    return poly_error, poly_w


# K folds Cross Validation and Grid Search

def K_folds_cross_validation(X_train, X_test, y_train, y_test, model_func, test_func, folds=5, param=None):
    '''
    Computes k folds cross validation and returns best model 
    '''
    #Create splits 
    fold_subsets = k_fold_splits(X_train, y_train, folds)

    #Set up best model and error for cross val comparison 
    best_err = float('inf')
    best_w = None 


    #Execute Cross Validation and Return best model 
    for train_X, train_y, subset_X, subset_y in fold_subsets:
        error, w_hat = model_func(train_X, subset_X, train_y, subset_y, param)
        #print("The error is ", error)
        if error < best_err:
            #print("old best err: {}; new best err {}".format(best_err, error))
            best_err = error
            best_w = w_hat

    #Use best weights to make predictions on the test data 
    X_test = np.insert(X_test, 0, 1, axis=1) 
    cv_err = test_func(X_test, y_test, best_w)
    
    return best_w, cv_err


def param_search(X_train, X_test, y_train, y_test, model_func, test_func, params=[], cv_func=None, folds=None):
    """
    Performs grid search for best parameter (includes cross validation if necessary)
    """ 
    best_err = float('inf')
    best_w = None 
    best_param = None 

    #if not also doing cross validation only apply the given model function to find best weights
    if not cv_func: 
        for param in params:
            error, w_hat = model_func(X_train, X_test, y_train, y_test, param)
            print("for param {}, the error is {}".format(param, error))
            if error < best_err:
                best_err = error
                best_w = w_hat
                best_param = param

    #if using cross validation, call cross val function instead and apply each parameter for each cross val step 
    else:
        for param in params:
            w_hat, error = cv_func(X_train, X_test, y_train, y_test, model_func, test_func, folds=5, param=param)
            print("for param {}, the error is {}".format(param, error))
            if error < best_err:
                best_err = error
                best_w = w_hat
                best_param = param

    X_test = np.insert(X_test, 0, 1, axis=1) 
    param_err = test_func(X_test, y_test, best_w)

    return param_err, best_w, best_param 


#State specific splitting and validation test code 

def gen_nonca_state_codes_list(nonca_set):
    '''
    generates list of unique state codes to index for cross-validation by state
    '''
    nonca_state_codes = []
    for (cname, cdata) in nonca_set['state_code'].iteritems():
        if cdata not in nonca_state_codes:
            nonca_state_codes.append(cdata)

    return nonca_state_codes


def cvsplit_by_state(nonca_set):
    '''
    generates test/train sets by state
    input: 
        nonca_set: pandas df with complete merged county data, excluding CA counties
    '''
    nonca_state_codes = gen_nonca_state_codes_list(nonca_set)
    sets = []
    for i in nonca_state_codes:
        subset = nonca_set.loc[nonca_set['state_code'] == i]
        subset_X = subset.drop('Deaths involving COVID-19', axis=1)
        subset_y = subset[['Deaths involving COVID-19']]

        train = nonca_set.loc[nonca_set['state_code'] != i]
        train_X = train.drop('Deaths involving COVID-19', axis=1)
        train_y = train[['Deaths involving COVID-19']]
        
        #convert pd dfs to np with df.to_numpy()
        subset_y = subset_X.to_numpy(dtype=float)
        subset_X = subset_X.to_numpy(dtype=float)
        train_X = train_X.to_numpy(dtype=float)
        train_y = train_y.to_numpy(dtype=float)
        
        sets.append([train_X, train_y, subset_X, subset_y])
    
    return sets


def by_states_cross_validation(nonca_set, ca_set_labels, ca_set_features, model_func, test_func, param=None):
    '''
    Computes k folds cross validation and returns best model 
    '''
    #Create splits 
    fold_subsets = cvsplit_by_state(nonca_set)

    #Set up best model and error for cross val comparison 
    best_err = float('inf')
    best_w = None 
     
    #Execute Cross Validation and Return best model 
    for train_X, train_y, subset_X, subset_y in fold_subsets:
        error, w_hat = model_func(train_X, subset_X, train_y, subset_y, param)
        print("The error is ", error)
        if error < best_err:
            print("old best err: {}; new best err {}".format(best_err, error))
            best_err = error
            best_w = w_hat
    
    ca_set_labels = ca_set_labels.to_numpy(dtype=float)
    ca_set_features = ca_set_features.to_numpy(dtype=float)
    #Use best weights to make predictions on the test data 
    X_test = np.insert(ca_set_features, 0, 1, axis=1) 
    cv_err = test_func(ca_set_features, ca_set_labels)
    
    return best_w, cv_err




