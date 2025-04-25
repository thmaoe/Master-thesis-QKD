## Function for simulating different protocols and output statistics

import qutip as qt
from math import sqrt, exp
import numpy as np

def getMeasOps(): #useful for third reproducing paper results
    N=20 #truncation of fock space
    M0 = qt.tensor((qt.qeye(N) - qt.fock(N,0)*qt.fock(N,0).dag()), qt.qeye(N)) #click in first
    M1 = qt.tensor(qt.qeye(N), (qt.qeye(N) - qt.fock(N,0)*qt.fock(N,0).dag())) #in second
    Mboth = qt.tensor((qt.qeye(N) - qt.fock(N,0)*qt.fock(N,0).dag()), (qt.qeye(N) - qt.fock(N,0)*qt.fock(N,0).dag())) #in both
    M0_only = M0 - Mboth
    M1_only = M1 - Mboth
    Minc = qt.tensor(qt.qeye(N), qt.qeye(N)) - M0_only - M1_only - Mboth

    return M0_only, M1_only, Mboth, Minc

def measure(p, eff=1, impl='1'):
    if impl == '1':
        if np.random.rand() < p[1]:
            if np.random.rand() < eff:
                return 1
            else:
                return 2
        else:
            if np.random.rand() < 1e-6: # dark counts
                return 1
            else:
                return 2
        
    elif impl == '2':

        r = np.random.rand() #for measurement
        reff = np.random.rand() #for efficiency

        if r < p[0]:
            if reff < eff:
                return 0
            else:
                return 2
        elif r < p[0] + p[1]:
            if reff < eff:
                return 1
            else:
                return 2
            
        else:
            if np.random.rand() < 1e-6: #dark counts
                if np.random.rand() < 1/2:
                    return 0
                else:
                    return 1
            else:
                return 2       
             
    elif impl == '3':
        p0 = p[0]
        p1 = p[1]
        pboth = p[2]

        r = np.random.rand()
        reff = np.random.rand()

        if r<p0:
            if reff < eff:
                return 0
            else:
                return 2
        elif r < p0 + p1:
            if reff < eff:
                return 1
            else:
                return 2
        elif r < p0 + p1 + pboth:
            if np.random.rand() < 1/2:
                if reff < eff:
                    return 0
                else:
                    return 2
            else:
                if reff < eff:
                    return 1
                else:
                    return 2
        else:
            if np.random.rand() < 1e-6:
                if np.random.rand()<1/2:
                    return 0
                else:
                    return 1
            else:
                return 2
                  
    else:
        return "Wrong input for implementation" 
        

def pick_x(px1):
    if np.random.rand() < px1:
        return 1
    else:
        return 0


def get_stat(data, impl, deadtime=False):
    if impl == '1' or impl == '2':
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

        for x in (0, 1):
            N_total = total[x]
            N_click = counts[0][x] + counts[1][x]

            if N_total == 0:
                continue

            click_rate = (N_click / N_total) * pulse_rate

            if deadtime:
                cd = 1 / (1 + td * click_rate)
            else: 
                cd = 1

            p0x = p[0][x] * cd
            p1x = p[1][x] * cd
            p2x = 1 - (p0x + p1x)  # normalize

            p[0][x] = p0x
            p[1][x] = p1x
            p[2][x] = p2x

        return p, counts
    
    else:
        counts = {0: {0: 0, 1: 0, 2: 0}, 1: {0: 0, 1: 0, 2: 0}, 2: {0: 0, 1: 0, 2: 0}}
        total = {0: 0, 1: 0, 2: 0}

        for b, x in data:
            counts[b][x] += 1
            total[x] += 1

        p = {
            b: {
                x: (counts[b][x] / total[x]) if total[x] > 0 else 0
                for x in (0, 1, 2)
            }
            for b in (0, 1, 2)
        }

        return p, counts

def getProbas(alpha, eff, dc, impl, deadtime = False): #theoretical p(b|x)
    if impl == '1':
        p10 = dc
        p20 = 1 - dc
        p11 = 1 - (1 - dc)*exp(-abs(alpha)**2*eff)
        p21 = (1 - dc)*exp(-abs(alpha)**2*eff) ##prob no dc * get no click (|<0|alpha>|^2), and eff = BS of transmissivity eta so |alpha> becomes |alpha*sqrt(eta)>
        delta = exp(-abs(alpha)**2/2)
        p = {0: {0: 0.0, 1: 0.0}, 1: {0: p10, 1: p11}, 2: {0: p20, 1: p21}}

    elif impl == '2':
        p00 = 1 - (1 - dc)*exp(-abs(alpha)**2*eff)
        p10 = (1 - p00)*dc #no click in first * dark count 
        p11 = 1 - (1 - dc)*exp(-abs(alpha)**2*eff)
        p01 = (1 - p11)*dc
        p20 = (1 - dc)**2*exp(-abs(alpha)**2*eff)
        p21 = (1 - dc)**2*exp(-abs(alpha)**2*eff)
        p = {0: {0: p00, 1: p01}, 1: {0: p10, 1: p11}, 2: {0: p20, 1: p21}}
        delta = exp(-abs(alpha)**2)
    else:
        return "Wrong impl number"
    
    if deadtime:
        td = 34e-9  # detector dead time in seconds
        pulse_rate = 50e6  # pulse rate in Hz

        for x in (0, 1):
            p0x = p[0][x]
            p1x = p[1][x]

            click_rate = (p0x + p1x) * pulse_rate

            cd = 1 / (1 + td * click_rate)

            p0x_corr = p0x * cd
            p1x_corr = p1x * cd
            p2x_corr = 1 - (p0x_corr + p1x_corr)  # renormalize

            p[0][x] = p0x_corr
            p[1][x] = p1x_corr
            p[2][x] = p2x_corr

    return delta, p



def doSimul(alpha, px1=1/2, impl='1', nPoints=100000, eff=1, deadtime=False, badSource = False):

    if badSource:
        goodAlpha = alpha

    if impl == '3':
        N=20 #Fock space truncation
        psi0 = qt.tensor(qt.coherent(N, alpha[0]), qt.coherent(N, 0))
        psi1 = qt.tensor(qt.coherent(N, 0), qt.coherent(N, alpha[0]))
        psi2 = qt.tensor(qt.coherent(N, alpha[1]), qt.coherent(N, alpha[1]))
        MeasOps = getMeasOps()
        p00 = abs((psi0.dag() * MeasOps[0] * psi0).real)
        p10 = abs((psi0.dag() * MeasOps[1] * psi0).real)
        pboth0 = abs((psi0.dag() * MeasOps[2] * psi0).real)
        p01 = abs((psi1.dag() * MeasOps[0] * psi1).real)
        p11 = abs((psi1.dag() * MeasOps[1] * psi1).real)
        pboth1 = abs((psi1.dag() * MeasOps[2] * psi1).real)
        p02 = abs((psi2.dag() * MeasOps[0] * psi2).real)
        p12 = abs((psi2.dag() * MeasOps[1] * psi2).real)
        pboth2 = abs((psi2.dag() * MeasOps[2] * psi2).real)

        p = [(p00, p10, pboth0), (p01, p11, pboth1), (p02, p12, pboth2)]
    else:
        pass
    
    if impl == '1' or impl == '2':

        data = []
        alphas = []
        i = 0
        for _ in range(nPoints):
            if badSource:
                alpha = goodAlpha + np.random.normal(0, 0.05/(2*goodAlpha)) #the source is not perfect so sends |alpha + delta>
                alphas.append(alpha)
            x = pick_x(px1)
            _, p = getProbas(alpha, 1, 0, '1', False)
            p = [p[b][x] for b in range(3)]
            b = measure(p, eff, impl)
            data.append((b,x))
        
        if impl == '1' : 
            if badSource:
                alphamax = np.percentile(np.abs(alphas), 99.9)
                delta = exp(-abs(alphamax)**2/2)
            else:
                delta = exp(-abs(alpha)**2/2)
        else: 
            if badSource:
                alphamax = np.percentile(np.abs(alphas), 99.9)
                delta = exp(-abs(alphamax)**2)
            else:
                delta = exp(-abs(alpha)**2)

        probs = get_stat(data, impl, deadtime)

        if badSource:
            return alphamax, delta, probs
        else:
            return delta, probs
    
    else:

        d01 = abs(psi0.dag() * psi1)
        d02 = abs(psi0.dag() * psi2)
        d12 = abs(psi1.dag() * psi2)

        data = []

        for _ in range(nPoints):
            r = np.random.rand()
            if r < 1/3:
                x = 0
            elif r < 2/3:
                x = 1
            else:
                x = 2
            
            if x == 0:
                b = measure(p[0], eff, impl)
                data.append((b,x))
            elif x == 1:
                b = measure(p[1], eff, impl)
                data.append((b,x))
            else:
                b = measure(p[2], eff, impl)
                data.append((b,x))
            
        probs = get_stat(data, impl)

        return (d01, d02, d12), probs

    

