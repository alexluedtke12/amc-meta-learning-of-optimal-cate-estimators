#!/usr/bin/env python3

gam = True
M = 20
n = 500
s = 1
wdim = 10
niter = 5000000

import generic_experiment

generic_experiment.run_experiment(gam,M,n,s,wdim,niter)
