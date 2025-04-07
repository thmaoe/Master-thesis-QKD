## Function for simulating different protocols and output statistics

import qutip as qt
from math import sqrt
import numpy as np

def getMeasOps(impl='1'):
    N=10 #truncation of fock space

    if impl == '1':
        M = qt.qeye(N) - qt.fock(N,0) * qt.fock(N,0).dag()
        return M
    elif impl == '2':
        M0 = qt.tensor((qt.qeye(N) - qt.fock(N,0)*qt.fock(N,0).dag()), qt.qeye(N)) #Click is photon in first dimension
        M1 = qt.tensor(qt.qeye(N), (qt.qeye(N) - qt.fock(N,0)*qt.fock(N,0).dag())) #in second dimension
        Minc = qt.tensor(qt.qeye(N), qt.qeye(N)) - M0 - M1 #no click

        return M0, M1, Minc
    else:
        return "Wrong input for implementation"

def measure(state, eff=1, impl='1'):
    if impl == '1':
        MeasOps = getMeasOps(impl)
        prob = abs((state.dag() * MeasOps * state).real)
        if np.random.rand() < prob:
            if np.random.rand() < eff:
                return 1
            else:
                return 2
        else:
            return 2
        
    elif impl == '2':
        MeasOps = getMeasOps(impl)
        p0 = abs((state.dag() * MeasOps[0] * state).real)
        p1 = abs((state.dag() * MeasOps[1] * state).real)

        r = np.random.rand() #for measurement
        reff = np.random.rand() #for efficiency

        if r < p0:
            if reff < eff:
                return 0
            else:
                return 2
        elif r < p0 + p1:
            if reff < eff:
                return 1
            else:
                return 2
        else:
            return 2
        
    else:
        return "Wrong input for implementation" 
        

def pick_x(px1):
    if np.random.rand() < px1:
        return 1
    else:
        return 0

def get_stat(data): ##Data is a list of (b,x) tuples with b=2 with b was inconclusive
    # Initialize count dictionaries
    counts = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0, 1: 0}}
    total = {0: 0, 1: 0}

    # Count occurrences
    for b, x in data:
        counts[b][x] += 1
        total[x] += 1

    # Compute probabilities, avoiding division by zero
    p = {
        b: {
            x: (counts[b][x] / total[x]) if total[x] > 0 else 0
            for x in (0, 1)
        }
        for b in (0, 1, 2)
    }

    return p, counts

def doSimul(alpha, px1, impl='1', nPoints=100000, eff=1):

    N=10 #Fock space truncation
    if impl == '1':
        psi0 = qt.coherent(N, 0)
        psi1 = qt.coherent(N, alpha)
    elif impl == '2':
        psi0 = qt.tensor(qt.coherent(N, alpha), qt.coherent(N, 0))
        psi1 = qt.tensor(qt.coherent(N, 0), qt.coherent(N,alpha))
    else:
        return "Wrong implementation number"

    delta = abs((psi0.dag() * psi1)) #overlap

    data = []

    for _ in range(nPoints):
        x = pick_x(px1)
        if x:
            b = measure(psi1, eff, impl)
            data.append((b,x))
        else:
            b = measure(psi0, eff, impl)
            data.append((b,x))

    probs = get_stat(data)
    
    return delta, probs

