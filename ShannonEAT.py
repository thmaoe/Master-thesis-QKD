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
        h = shannonLower.runOpti(delta, p, px)
        print(h)
        Hs.append(h)

    Hmin = min(Hs)
    var = np.std(Hs)
    fmin = Hmin - var - d
    print(fmin)

    g = -np.log(1-np.sqrt(1-eps**2))
    V = np.log(2*nB**2 + 1) + np.sqrt(2)
    K = (2 - alpha)**3/(6*(3-2*alpha)**3*np.log(2)) * 2**((alpha-1)/(2-alpha)*(2*np.log(nB)))*np.log(2**(2*np.log(nB)) + np.e**2)**3

    print((alpha - 1)/(2 - alpha)*np.log(2)/2*V**2)
    print(1/N * g/(alpha-1))
    print(((alpha-1)/(2-alpha))**2 * K)
    res = fmin - (alpha - 1)/(2 - alpha)*np.log(2)/2*V**2 - 1/N * g/(alpha-1) - ((alpha-1)/(2-alpha))**2 * K
    return res

