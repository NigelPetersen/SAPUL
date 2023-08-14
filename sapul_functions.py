
import numpy as np
import pandas as pd
import scipy as sp
import random
from random import sample
from sklearn import metrics

# MISCELLANEOUS FUNCTIONS

def repeat_row(n):
    return np.reshape(np.repeat(np.array(list(range(1,n+1))), n), (n,n))

def repeat_col(n):
    return np.transpose(repeat_row(n))

def autocorrelation_matrix(p=100, rho=0.3):
    """
    Given a dimension p and correlation coefficient rho, return the order 1 autoregressive correlation matrix A with entries A[i,j] = rho**(|i-j|)
    """
    return rho**(abs(repeat_row(p) - repeat_col(p)))

def expit(x):
    return np.exp(x)/(1+np.exp(x))

def Binomial_log_likelihood(beta, data):
    """
    Given a vector of coefficients beta and a data matrix data, return the log-likelihood of the logistic model y|x ~ Binomial(expit(beta^Tx))
    """
    n = data.shape[0]
    y = data[:,0]
    X = data[:,1:]
    L = 0
    pi = expit(np.matmul(X, beta))
    return np.dot(y, mp.log(pi)) + np.dot(np.ones(n)-y, np.log(np.ones(n)-pi))


def Anchor_log_likelihood(params, data):
    """
    Given a vector params of regression coefficients and a weight c, and a data matrix, return the negative log-likelihood of the
    weighted logistic model y|x ~ Binomial(expit(c)*expit(beta^Tx))
    """
    return -Binomial_log_likelihood(expit(params[0])*params[1:], data )

def vector_to_matrix(v,n):
    """
    Given a vector v and a positive integer n, return a matrix with n rows each consisting of the vector v
    """
    return np.repeat(v.reshape((1,v.shape[0])) , n, axis=0)

# DATA GENERATION

 def generate_data(n=100, N=20000, p=50, prev=0.2, x_cov=0.4, b0=0, g0=0, corr=False, col=1):
    # n = number of observed labels (positive)
    # N = number of unobserved labels
    # p = number of covariates
    # prev = phenotype prevalence
    # x_cov = autocorrelation param to generate covariance matrix
    # b0 = initial value for beta
    # g0 = initial value for gamma
    # corr = indicator to violate independence assumption.
    # col = column of design matrix to add to surrogate when corr==True
    """
    Given the above parameters, return a matrix of data consisting of true (unobserved) labels, observed (positive-only) labels,
    observed features and observed surrogate features.
    """

    if b0 == 0:
        beta = np.concatenate((np.array([-0.6, 0.6, 0.3, -0.3, 0.3]), np.zeros(p-5)), axis=0)
    if g0 == 0:
        gamma = 1.5
    N_total = n + N
    y = np.random.binomial(1, prev, N_total)
    x_mean = np.ones((p, N_total)) + (y*vector_to_matrix(beta, N_total)).transpose()
    x = np.random.multivariate_normal(cov = autocorrelation_matrix(p, x_cov), size = N_total) + x_mean
    
    s_mean = np.ones((N_total,1)) + gamma*y
    s = np.random.normal(loc=s_mean, size = N_total)

    if corr:
        s += x[:,col-1]

    obs_ind = sample(np.where(y==1)[0],n)
    a = np.zeros(len(y))
    a[obs_ind] = 1
    return np.concatenate((y,a,x,s),axis=1)







