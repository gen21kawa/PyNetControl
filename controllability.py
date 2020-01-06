"""
Python implementation of average controllability and modal controllability,
which is adapted from MATLAB code written by Complex Systems Lab at UPenn
https://complexsystemsupenn.com/codedata
"""


import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import numpy.matlib

def ave_control(A):
    """
    Input: 
      - A, an adjacency matrix
    Output:
      - values, average controllability of each node in A. 
    Adapted from MATLAB code by Complex Systems Lab at UPenn.
    """
    # normalize matrix
    u,s,_ = np.linalg.svd(A)
    ss = max(s)
    A = A/(1+ss)
    T,Z = scipy.linalg.schur(A,'real')
    midMat = np.transpose((Z**2))
    v = np.diag(T)
    v.shape = (len(v),1)
    temp = np.matmul(v,np.transpose(v))
    a = np.diag(np.ones(np.shape(temp))-temp)
    a.shape = (len(v),1)
    msize = np.size(A,1)
    P = np.matlib.repmat(a,1,msize)
    values = np.transpose(sum(midMat/P))
    return values
    
def modal_control(A):
    """
    Input: 
      - A, an adjacency matrix
    Output:
      - phi, modal controllability of each node in A. 
    Adapted from MATLAB code by Complex Systems Lab at UPenn.
    """
    u,s,_ = np.linalg.svd(A)
    ss = max(s)
    A = A/(1+ss)
    T,Z = scipy.linalg.schur(A,'real')
    eigVals = np.diag(T)
    N = np.size(A,1)
    phi = np.zeros((N,1))
    for i in range(N):
        phi[i] = np.matmul(Z[i][:]**2, (np.ones(np.shape(eigVals))-eigVals**2))
    return phi
    
