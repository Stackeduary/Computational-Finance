# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:11:12 2020

@author: Bill
"""

import numpy as np
from scipy import linalg

def asian_solver(ns, nI, m, Smax, Imax, r, D, T, sigma, p, phi1, phi2):
    """sigma is assumed to be a function of s and t
    phi1, phi2 are functions of s, Imax, t and Smax, I, t
    solve in the region 
    0 <= s <= Smax
    0 <= I <= Imax
    0 <= t <= T
    """
    
    s = np.linspace(0, Smax, ns+1)
    ds = Smax/ns
    I = np.linspace(0, Imax, nI+1)
    dI = Imax/nI
    t = np.linspace(0, T, m+1)
    dt = T/m
    
    V = np.zeros(shape = (ns+1, nI+1, m+1))
    
    # fill in the final condition
    # simplest approach is to use a for-cycle for one index
    # for the other index, use vector operations
    for i in range(0, ns+1) :
        V[i, :, m] = p(s[i], I/T)
        
    # in the formulas for coefficients we use indexes i from 1 to ns-1
    # define a vector of indexes before starting for cycles
    i = np.arange(1, ns)
    
    # define matrix M
    M = np.zeros(shape = (ns+1, ns+1))
    
    M[0, 0] = 1
    M[ns, ns] = 1
    
    rho = dt/ds**2
    
    # define vector F
    F = np.zeros(ns+1)
    
    for k in range(m-1, -1, -1):
        
        # Fill in the values for I=Imax
        V[:, nI, k] = phi1(s, Imax, t[k])

        M[i, i-1] = rho/2*(-s[i]**2*sigma(s[i], t[k])**2 + (r - D)*s[i]*ds)
        M[i, i] = 1 + rho*s[i]**2*sigma(s[i], t[k])**2 + s[i]*dt/dI + r*dt
        M[i, i+1] = -rho/2*(s[i]**2*sigma(s[i], t[k])**2 + (r - D)*s[i]*ds)
        
        # All other values for j=nI-1,nI-2,...,0
        for j in range(nI-1, -1, -1):
            # Fill F
            F[0] = p(0, I[j]/T)*np.exp(-r*(T - t[k]))
            F[ns] = phi2(Smax, I[j], t[k])
            F[i] = s[i]*dt/dI*V[i, j+1, k] + V[i, j, k+1]
            # solve the system
            V[:, j, k] = linalg.solve(M, F)
    # return V[:, 0, 0]
    return V

r = .02
D = .02
T = .4
S0 = 12.5
Imax = 84
Smax = 100
m = 10
ns = 16
nI = 8

def sigma(s, t):
    return 0.6

def p(s, A):
    return np.maximum(.5*s + .3*A + 3, 0)

c1 = 3
c2 = .5
c3 = .3/T

def v_spec(s, I, t):
    return c1*np.exp(-r*(T-t)) + np.exp(-r*(T-t))*(c2 + c3*(T-t))*s + c3*np.exp(-r*(T-t))*I

def phi1(s, Imax, t):
    return np.maximum(v_spec(s, Imax, t), 0)

def phi2(Smax, I, t):
    return np.maximum(v_spec(Smax, I, t), 0)

V = asian_solver(ns, nI, m, Smax, Imax, r, D, T, sigma, p, phi1, phi2)

# indexs when S0 = 12.5 is Smax/S0, which equals 100/12.5 = 8
indexs = np.int64(ns*S0/Smax)
print(indexs)

# when t = 0, I also equals 0, so answer1 is at V[ns//indexs, 0, 0]
# answer1 = V[indexs, 0, 0]
answer1 = V[indexs, 0, 0]


# indexS when S(t = .32) = 87.5 is S(t = .32)/dS = 87.5/6.25 since ds = 100/16 = 6.25
dS = Smax/ns
# indexS = np.int64(ns*87.5/Smax)
indexS = np.int64(87.5/(Smax/ns))
print(indexS)
# 14

# indexI when I(t = .32) = 31.5 is I(t = .32)/dI = 31.5/10.5 since dI = 84/8 = 10.5
dI = Imax/nI
indexI = 31.5/dI
print(indexI)
# 3

# indext when t = .32 is .32/dt = 8 since dt = .04
dt = T/m
indext = .32/dt
print(indext)
# 8

# indexS = np.int64(87.5/(Smax/ns))
indexI = np.int64(31.5/(Imax/nI))
indext = np.int64(.32/T/m)

# answer2 = V[indexS, indexI, indext]
answer2 = V[14, 3, 8]

print(answer1, answer2)