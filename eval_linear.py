#!/usr/bin/env python

import numpy as np
import torch
import pandas as pd
import os

import learn2predict as l2p

wdim = 10
n = 100
gam = False

li = []
for setting in [1,2,3,4]:
    for s in [1,4,7,10]:
        print(s)
        Pi, Pi_opt, Pi_sched = l2p.initPi(s,s+2,wdim,gam=gam,lr=0.001)
        T, T_opt, T_sched = l2p.initT(lr=0.001,rank_based=False,ablation=0)
        fn_main = 'estimators/Linear_n'+str(n)+'_s'+str(s)+'_wdim'+str(wdim)
        iteration, loss_list = l2p.load_model(T, T_opt, T_sched, Pi, Pi_opt, Pi_sched, fn_main+'.tar', fl_backup = fn_main+'_backup.tar')

        PiOWL = l2p.PriorOWL(wdim,setting=setting).to(l2p.device)
        risk, se = l2p.interrogate(T,PiOWL,n,100,numbatch=10000,batchsize=100)
        li.append(pd.DataFrame(data={'setting': [setting], 's': [s],'MSE': [risk.item()],'se': [se.item()]}))
    df_out = pd.concat(li, axis=0, ignore_index=True)
    df_out.to_csv(r'linear_performance/amc.csv',index=False)


