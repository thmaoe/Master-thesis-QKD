### simulation.py contains code to generate statistics for various versions of the protocol.
- impl = '0' -> photon number, '1' -> time bin, '2' -> bpsk, '3' -> modified bpsk
- measure(p, eff, impl): gives result b of one round of the experiment given probas p(b|x), efficiency of detector eff and implementation used.
- get_stat(data, deadtime): gives the statistics for multiple rounds of experiment, given data a list of results (b,x) and possibility to apply the deadtime of detector.
- getProbas(alpha, eff, dc, impl, deadtime, d): gives theoretical probas for various implementations
- doSimul(alpha, px1, impl, nPoints=100000, eff, deadtime, badSource, d): gives stats for nPoints round of experiment, with wanted parameters.
