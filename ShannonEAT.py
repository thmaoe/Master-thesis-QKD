## functions to compute lower bound on smooth min entropy using EAT
import shannonLower
import statistics
import numpy as np

def getH(ps, N, delta, px):
    eps = 1e-8
    nB = 3
    d = 1e-3
    alpha = 1 + 1/np.sqrt(N)

    Hs = []
    for p in ps:
        h, lambdas, rs, cm = shannonLower.getHDual(delta, p, px)
        Hs.append(h)

    Min = np.min(Hs)
    Var = np.var(Hs)
    Max = np.max(Hs)
    
    res = []
    for n in range(len(ps)):

        g = -np.log(1-np.sqrt(1-eps**2))
        V = np.log(2*nB**2 + 1) + np.sqrt(2 + Var)
        K = (2 - alpha)**3/(6*(3-2*alpha)**3*np.log(2)) * 2**((alpha-1)/(2-alpha)*(2*np.log(nB)+Max-Min))*np.log(2**(2*np.log(nB)+Max-Min) + np.e**2)**3

        h = Hs[n] - (alpha - 1)/(2 - alpha)*np.log(2)/2*V**2 - 1/N * g/(alpha-1) - ((alpha-1)/(2-alpha))**2 * K
        res.append(h)
    
    return np.mean(res)

def getH2(ps, N, delta, px):
    eps = 1e-8
    nB = 3
    d = 1e-3
    alpha = 1 + 1/np.sqrt(N)

    Hs = []
    for p in ps:
        h = shannonLower.runOpti(delta, p, px)
        Hs.append(h)

    Hmin = min(Hs)
    var = np.std(Hs)
    fmin = Hmin - var - d

    g = -np.log(1-np.sqrt(1-eps**2))
    V = np.log(2*nB**2 + 1) + np.sqrt(2)
    K = (2 - alpha)**3/(6*(3-2*alpha)**3*np.log(2)) * 2**((alpha-1)/(2-alpha)*(2*np.log(nB)))*np.log(2**(2*np.log(nB)) + np.e**2)**3

    res = fmin - (alpha - 1)/(2 - alpha)*np.log(2)/2*V**2 - 1/N * g/(alpha-1) - ((alpha-1)/(2-alpha))**2 * K
    return res

def getH3(ps, N, lambdas, Rs, cm):
    eps = 1e-8
    nB = 3
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
    
    res = []
    for n in range(len(ps)):

        g = -np.log(1-np.sqrt(1-eps**2))
        V = np.log(2*nB**2 + 1) + np.sqrt(2 + Var)
        K = (2 - alpha)**3/(6*(3-2*alpha)**3*np.log(2)) * 2**((alpha-1)/(2-alpha)*(2*np.log(nB)+Max-Min))*np.log(2**(2*np.log(nB)+Max-Min) + np.e**2)**3

        h = Hs[n] - (alpha - 1)/(2 - alpha)*np.log(2)/2*V**2 - 1/N * g/(alpha-1) - ((alpha-1)/(2-alpha))**2 * K
        res.append(h)
    
    return np.mean(res)
