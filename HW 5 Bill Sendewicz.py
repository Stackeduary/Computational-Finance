# Bill Sendewicz
# 5/5/2020
# Feliz Cinco de Mayo	

import numpy as np
from scipy import linalg

def sigma(s, t):
    return .3 + (.3*np.exp(-t/2)/(1 + .1*(s - 30)**2))

r = .02
D = .02
T = .6
S0 = 30
xmin = 0
xmax = 75
n = 60
m = 5
accuracy = .000087

def alpha(x, t):
    return (x*sigma(x, t))**2/2

def beta(x, t):
    return (r - D)*x

def p(s):
    return (19/20*s)*(s <= 20) + (19*(s/20 - 5*(s/20 - 1)**2 + 3.1*(s/20 - 1)**3))*(s > 20)*(s <= 40) + 1.9*(s/20 - 1)*(s > 40)

def u0(x):
    return p(x)

def phi1(xmin, t):
    return p(0)*np.exp(-r*(T - t))

def phi2(xmax, t):
    return p(xmax)

def crankNicolson(m, n, xmin, xmax, T, phi1, phi2):
    dx = xmax/n
    dt = T/m
    x = np.linspace(0, xmax, n+1)
    t = np.linspace(0, T, m+1)
    U = np.zeros(shape = (n+1, m+1))
    
    U[:, m] = u0(x)
    M = np.zeros(shape = (n+1, n+1))
    F = np.zeros(n+1)
    M[0, 0] = 1
    M[n, n] = 1
    
    i = np.arange(1, n)
    
    for k in range(m-1, -1, -1):
        aik = -dt/(2*dx**2)*(alpha(x[i], t[k]) - beta(x[i], t[k])/2*dx)
        bik = 1 + dt/dx**2*alpha(x[i], t[k]) + r*dt/2
        cik = -dt/(2*dx**2)*(alpha(x[i], t[k]) + beta(x[i], t[k])/2*dx)

        M[i, i-1] = aik
        M[i, i] = bik
        M[i, i+1] = cik

        dik = dt/(2*dx**2)*(alpha(x[i], t[k+1]) - beta(x[i], t[k+1])/2*dx)
        fik = dt/(2*dx**2)*(alpha(x[i], t[k+1]) + beta(x[i], t[k+1])/2*dx)
        eik = 1 - dt/dx**2*alpha(x[i], t[k+1]) - r*dt/2

        F[0] = phi1(xmin, t[k])
        F[n] = phi2(xmax, t[k])
        
        F[i] = dik*U[i-1, k+1] + eik*U[i, k+1] + fik*U[i+1, k+1]
    
        U[:, k] = linalg.solve(M, F)
    return U[:, 0]


error = accuracy + 1
prices = crankNicolson(m, n, xmin, xmax, T, phi1, phi2)
priorResult = prices[n//2]

while (error > accuracy):
    n = n*2
    m = m*2
    answer = crankNicolson(m, n, xmin, xmax, T, phi1, phi2)[n//2]
    error = np.abs(priorResult - answer)/3
    priorResult = answer