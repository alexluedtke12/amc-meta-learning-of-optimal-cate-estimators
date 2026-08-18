#!/usr/bin/env python3

import numpy as np
import torch
import pandas as pd
import os

import learn2predict as l2p

wdim = 10
n = 500
gam = True
s = 4
M = 10

li = []
for setting in [1,2,3,4]:
    print([setting,s,M],flush=True)
    Pi, Pi_opt, Pi_sched = l2p.initPi(s,s+2,wdim,M=M,gam=gam,lr=0.005)
    T, T_opt, T_sched = l2p.initT(lr=0.001,rank_based=True,ablation=0)
    fn_main = 'estimators/Gam_m'+str(M)+'_n'+str(n)+'_s'+str(s)+'_wdim'+str(wdim)
    if os.path.exists(fn_main+'.tar'):
        iteration, loss_list = l2p.load_model(T, T_opt, T_sched, Pi, Pi_opt, Pi_sched, fn_main+'.tar', fl_backup = fn_main+'_backup.tar')

        PiOWL = l2p.PriorOWL(wdim,setting=setting).to(l2p.device)
        risk, se = l2p.interrogate(T,PiOWL,n,10000,numbatch=100,batchsize=100)
        li.append(pd.DataFrame(data={'setting': [setting],'M': [M], 's': [s],'MSE': [risk.item()],'se': [se.item()]}))
    df_out = pd.concat(li, axis=0, ignore_index=True)
    df_out.to_csv(r'gam_performance/amc.csv',index=False)


