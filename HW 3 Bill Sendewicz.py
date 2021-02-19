# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:52:35 2020
@author: Bill
"""

import numpy as np
from scipy import linalg

def f(x):
    return(-3)

def hw3(n):
    # boundary conditions
    a = 2
    b = 5
    
    # interval
    h = (b - a)/n
    
    x = np.linspace(a, b, n+1)
    i = np.arange(1, n)
    
    # F vector
    F = np.zeros(n+1)
    F[0] = 2
    F[n] = -1
    F[i] = f(x[i])

    # M matrix
    M = np.zeros(shape = (n+1, n+1))
    
    M[0,0] = 1
    M[n,n] = 1
        
    M[i, i] = (-2*3)/h**2 + (5 - x[i])
    
    M[i, i-1] = 3/h**2 - 2/(x[i]*2*h)
    
    M[i, i+1] = 3/h**2 + 2/(x[i]*2*h)

    y = linalg.solve(M, F)
    return (y)

print(hw3(10))