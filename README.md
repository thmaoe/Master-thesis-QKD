# Full code for my master's thesis

Not the best code but it does the job. It can be reused for other protocols by changing the different parameters in the code, it only needs to be feeded with the conditional
probabilities.

The figures in the report can be reproduced with the notebook results_paper.iypnb

- simulation.py contains code to get the probabilities for the various encodings studied, do monte carlo simulations of the protocol and simulate bad source.
- shannonLower.py allows to compute the lower-bound on Shannon entropy using work from BFF. Use function runOpti for the primal and getHDual for the dual.
- ShannonEAT.py computes the lower bound on the smooth min entropy using the GEAT. Use function getH3
- GraamBFF.py computes the lower bound on H(B|X,E) with our new technique combining the Gram NPA-like hierarchy with BFF results. The operators used can be changed directly in the code. Use runSDP function
- GraamBFF4.py, same but when B has an input
- HminLower.py computes lower-bound on Hmin, using the dual SDP for 2 outputs (conclusive/inconclusive). Use getHmin function
- HminLowerb.py, same but with 3 outputs (b=0,1,inconclusive)

