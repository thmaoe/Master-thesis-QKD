## functions to compute lower bound on smooth min entropy using EAT
import shannonLower
import statistics
import numpy as np

def getH3(ps, N, lambdas, Rs, cm, nB=3):
    eps = 1e-8
    d = 1e-3
    alpha = 1 + 1/np.sqrt(N)

    Hs = []
    for p in ps:
        h = 0.0
        for i,lambs in enumerate(lambdas):
            for b in range(3):
                for x in range(2):
                    h += lambs[b][x] * p[b][x]
            h += np.trace(Rs[i])
        h = cm - h
        Hs.append(h)


    Min = np.min(Hs)
    Var = np.var(Hs)
    Max = np.max(Hs)
    g = -np.log(1-np.sqrt(1-eps**2))
    V = np.log(2*nB**2 + 1) + np.sqrt(2 + Var)
    K = (2 - alpha)**3/(6*(3-2*alpha)**3*np.log(2)) * 2**((alpha-1)/(2-alpha)*(2*np.log(nB)+Max-Min))*np.log(2**(2*np.log(nB)+Max-Min) + np.e**2)**3
    
    res = Min - (alpha - 1)/(2 - alpha)*np.log(2)/2*V**2 - 1/N * g/(alpha-1) - ((alpha-1)/(2-alpha))**2 * K

    return res