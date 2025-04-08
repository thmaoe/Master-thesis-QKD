## Function for simulating different protocols and output statistics

import qutip as qt
from math import sqrt
import numpy as np

def getMeasOps(impl='1'):
    N=20 #truncation of fock space

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

def measure(p, eff=1, impl='1'):
    if impl == '1':
        if np.random.rand() < p:
            if np.random.rand() < eff:
                return 1
            else:
                return 2
        else:
            return 2
        
    elif impl == '2':

        p0 = p[0]
        p1 = p[1]

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


def get_stat(data, deadtime=False):
    td = 34e-9  # detector dead time in seconds
    pulse_rate = 50e6  # 50 MHz as in paper

    counts = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0, 1: 0}}
    total = {0: 0, 1: 0}

    for b, x in data:
        counts[b][x] += 1
        total[x] += 1

    p = {
        b: {
            x: (counts[b][x] / total[x]) if total[x] > 0 else 0
            for x in (0, 1)
        }
        for b in (0, 1, 2)
    }

    if deadtime:
        for x in (0, 1):
            N_total = total[x]
            N_click = counts[0][x] + counts[1][x]

            if N_total == 0:
                continue

            click_rate = (N_click / N_total) * pulse_rate

            cd = 1 / (1 + td * click_rate)

            p0x = p[0][x] * cd
            p1x = p[1][x] * cd
            p2x = 1 - (p0x + p1x)  # normalize

            p[0][x] = p0x
            p[1][x] = p1x
            p[2][x] = p2x

    return p, counts




def doSimul(alpha, px1, impl='1', nPoints=100000, eff=1, deadtime=False):

    N=20 #Fock space truncation
    if impl == '1':
        psi0 = qt.coherent(N, 0)
        psi1 = qt.coherent(N, alpha)
        MeasOps = getMeasOps(impl)
        prob0 = abs((psi0.dag() * MeasOps * psi0).real)
        prob1 = abs((psi1.dag() * MeasOps * psi1).real)
        p = [prob0, prob1]
    elif impl == '2':
        psi0 = qt.tensor(qt.coherent(N, alpha), qt.coherent(N, 0))
        psi1 = qt.tensor(qt.coherent(N, 0), qt.coherent(N,alpha))
        MeasOps = getMeasOps(impl)
        p00 = abs((psi0.dag() * MeasOps[0] * psi0).real)
        p10 = abs((psi0.dag() * MeasOps[1] * psi0).real)
        p01 = abs((psi1.dag() * MeasOps[0] * psi1).real)
        p11 = abs((psi1.dag() * MeasOps[1] * psi1).real)
        p = [(p00, p10), (p01, p11)]

    else:
        return "Wrong implementation number"

    delta = abs((psi0.dag() * psi1)) #overlap

    data = []

    Ndet = 0

    for _ in range(nPoints):
        x = pick_x(px1)
        if x:
            b = measure(p[1], eff, impl)
            if not b==2:
                Ndet += 1
            data.append((b,x))
        else:
            b = measure(p[0], eff, impl)
            if not b==2:
                Ndet += 1
            data.append((b,x))

    probs = get_stat(data, deadtime)
    
    return delta, probs

