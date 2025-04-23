##Function to compute lower bound on Shannon entropy

import cvxpy as cp
import numpy as np
import chaospy

def getMatricesFaster(xs, bs, impl=0): #Faster version -- better to use this one

    if impl==0:
        Mbs = {}

        for i in range(bs):
            Mbs[i] = cp.Variable((2,2), hermitian=True) ##we have only 2 inputs so we in dimension 2

        Xis = {}

        for b in range(bs):
            Xis[b] = {}
            for x in range(xs):
                Xis[b][x] = {}
                for bb in range(bs):
                    Xis[b][x][bb] = cp.Variable((2,2), complex = True)

        Thetas = {}
        for b in range(bs):
            Thetas[b] = {}
            for x in range(xs):
                Thetas[b][x] = {}
                for bb in range(bs):
                    Thetas[b][x][bb] = cp.Variable((2,2), complex = True) 

        return Mbs, Xis, Thetas
    else:
        Mbs = {}

        for i in range(bs):
            Mbs[i] = cp.Variable((3,3), hermitian=True) ##we have only 2 inputs so we in dimension 2

        Xis = {}

        for b in range(bs):
            Xis[b] = {}
            for bb in range(bs):
                Xis[b][bb] = cp.Variable((3,3), complex = True)

        Thetas = {}
        for b in range(bs):
            Thetas[b] = {}
            for bb in range(bs):
                Thetas[b][bb] = cp.Variable((3,3), complex = True) 

        return Mbs, Xis, Thetas

def getConstraintsFaster(Mbs, Xis, Thetas, rho, p, xs, bs, impl=0): 
    if impl == 0:
        constraints = []

        sumb_m = 0.0
        for b in range(bs):
            sumb_m += Mbs[b]
            for x in range(xs):
                sumb_xi = 0.0
                sumb_theta = 0.0
                for bb in range(bs): #We have Xi_bb^{xb}
                    G = cp.bmat([[Mbs[bb],           Xis[bb][x][b]], 
                                [Xis[bb][x][b], Thetas[bb][x][b]]])
                    constraints += [G >> 0.0] 

                    sumb_xi += Xis[bb][x][b]
                    sumb_theta += Thetas[bb][x][b]
                
                constraints += [sumb_xi == cp.trace(sumb_xi) * np.eye(2) / float(2)]
                constraints += [sumb_theta == cp.trace(sumb_theta) * np.eye(2) / float(2)]
                
        constraints += [sumb_m == np.eye(2)]

        for b in range(bs):
            for x in range(xs):
                constraints += [cp.trace(Mbs[b] @ rho[x]) == p[b][x]]
        
        return constraints
    
    else:
        constraints = []

        sumb_m = 0.0
        for bb in range(bs):
            sumb_m += Mbs[bb]
            sumb_xi = 0.0
            sumb_theta = 0.0
            for b in range(bs):
                sumb_xi += Xis[b][bb]
                sumb_theta += Thetas[b][bb]
                G = cp.bmat([[Mbs[b],           Xis[b][bb]], 
                            [Xis[b][bb], Thetas[b][bb]]])
                constraints += [G >> 0.0]

            
            constraints += [sumb_xi == cp.trace(sumb_xi) * np.eye(3) / float(3)]
            constraints += [sumb_theta == cp.trace(sumb_theta) * np.eye(3) / float(3)]
        
        constraints += [sumb_m == np.identity(3)]
                
        for b in range(bs):
            for x in range(xs):
                pred = cp.real(cp.trace(Mbs[b] @ rho[x]))
                constraints += [pred >= p[b][x] - 1e-3, pred <= p[b][x] + 1e-3]
        
        return constraints

def getHFaster(m, xs, bs, p, rho, w, t, px, impl = 0):
    if impl == 0:

        obj = 0.0
        for i in range(m-1):
            Mbs, Xis, Thetas = getMatricesFaster(xs, bs)
            constraints = getConstraintsFaster(Mbs, Xis, Thetas, rho, p, xs, bs)
            sumb = 0.0
            for b in range(bs):
                sumbb0 = 0.0
                sumbb1 = 0.0
                for bb in range(bs):
                    sumbb0 += Thetas[bb][0][b]
                    sumbb1 += Thetas[bb][1][b]
                sumb += (1-px) * cp.real(cp.trace(rho[0] @ (2*Xis[b][0][b] + (1 - t[i]) * Thetas[b][0][b] + t[i]*sumbb0)))
                sumb += px * cp.real(cp.trace(rho[1] @ (2*Xis[b][1][b] + (1 - t[i]) * Thetas[b][1][b] + t[i]*sumbb1)))

            subObj = sumb * (w[i]/(t[i]*np.log(2)))

            prob = cp.Problem(cp.Minimize(subObj), constraints)
            prob.solve(solver='MOSEK',verbose=False)
            obj += prob.value

        cm = 0.0

        for i in range(m-1):
            cm += w[i]/(t[i]*np.log(2))

        obj += cm

        return obj
    
    elif impl == 1:
        obj = 0.0
        for i in range(m):
            Mbs, Xis, Thetas = getMatricesFaster(xs, bs, impl=1)
            constraints = getConstraintsFaster(Mbs, Xis, Thetas, rho, p, xs, bs, impl=1)
            sumb = 0.0
            for b in range(bs):
                sumbb = 0.0
                for bb in range(bs):
                    sumbb += Thetas[bb][b]
                sumb += cp.real(cp.trace(rho[2] @ (2*Xis[b][b] + (1 - t[i]) * Thetas[b][b] + t[i]*sumbb)))
            subObj = sumb * (w[i]/(t[i]*np.log(2)))

            prob = cp.Problem(cp.Minimize(subObj), constraints)
            prob.solve(solver='MOSEK',verbose=False)
            obj += prob.value

        cm = 0.0

        for i in range(m):
            cm += w[i]/(t[i]*np.log(2))

        obj += cm

        return obj

    else:
        return "Wrong impl number"


def runOpti(delta, p, px, impl = 0):
    if impl == 0:
        rho0 = np.array([[1.,0],[0,0]])
        rho1 = np.array([[delta**2, delta*np.sqrt(1-delta**2)], 
                        [delta*np.sqrt(1-delta**2), 1-delta**2]])
        rho1 = rho1 / np.trace(rho1) #bc of floating errors

        rho = {0: rho0, 1: rho1}

        m_in = 4
        m = int(m_in*2)
        distribution = chaospy.Uniform(lower=0, upper=1)
        t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
        t = t[0]
        xs = 2
        bs = 3

        return getHFaster(m, xs, bs, p, rho, w, t, px)
        

    else:
        a = np.abs((delta[1]+delta[2])/np.sqrt(2.0*(1.0+delta[0])))**2.0 + np.abs((delta[1]-delta[2])/np.sqrt(2.0*(1.0-delta[0])))**2.0
        if a >= 1.0:
            a = 1.0
        if a <= 0.0:
            a = 0.0
        rho0 = np.array([[np.sqrt((1+delta[0])/2)], [np.sqrt((1-delta[0])/2)], [0.0]]) @ np.array([[np.sqrt((1+delta[0])/2), np.sqrt((1-delta[0])/2), 0.0]])
        rho1 = np.array([[np.sqrt((1+delta[0])/2)], [-np.sqrt((1-delta[0])/2)], [0.0]]) @ np.array([[np.sqrt((1+delta[0])/2), -np.sqrt((1-delta[0])/2), 0.0]])
        rho2 = np.array([[(delta[1] + delta[2])/np.sqrt(2*(1+delta[0]))], [(delta[1]-delta[2])/np.sqrt(2*(1-delta[0]))], [np.sqrt(1.0-a)]]) @ np.array([[(delta[1] + delta[2])/np.sqrt(2*(1+delta[0])), (delta[1]-delta[2])/np.sqrt(2*(1-delta[0])), np.sqrt(1.0-a)]])

        rho = {0: rho0, 1: rho1, 2: rho2}

        m_in = 4
        m = int(m_in*2)
        distribution = chaospy.Uniform(lower=0, upper=1)
        t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
        t = t[0]
        xs = 3
        bs = 3

        return getHFaster(m, xs, bs, p, rho, w, t, px, impl)
    



