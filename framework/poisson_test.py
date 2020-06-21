import sys
sys.path.append("../granger-busca")
sys.path.append("../framework")
import os
import numpy as np
from gb import simulate, GrangerBusca
import tick
import tick.hawkes as hk
from random_search import *
import matplotlib.pyplot as plt
from pp_metrics import *
from time import time
import pp_models

def hom_poisson(d, n_events):
    mu = np.ones(d)
    h = hk.SimuPoissonProcess(mu, max_jumps=d*n_events, verbose=False)
    h.simulate()
    return h.timestamps

def inhom_poisson(d, n_events, equal):
    mu = []
    for i in range(d):
        T = np.arange(0,10000,0.1, dtype=float)
        if equal:
            Y = np.sin(T/5)+1
        else:
            Y = np.sin(T/(i*2+1))+1
        tf = tick.base.TimeFunction((T, Y), dt=0.1)
        mu.append(tf)
    h = hk.SimuInhomogeneousPoisson(mu, max_jumps = d*n_events, verbose=False)
    h.simulate()
    return h.timestamps


def square_error(A_pred):
    return np.sum(A_pred**2)

def mu_se(baselines, baselinesGT=1):
    return np.mean((np.array(baselines)-baselinesGT)**2)

def save_model(modelName, adjacency, baselines, homogenous, sim):
    if homogenous:
        filepath = "results/hom_poisson/"+modelName+"/"
    else:
        filepath = "results/inhom_poisson/"+modelName+"/"

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    np.save(filepath+"/adjacency_"+str(sim), adjacency)
    np.save(filepath+"/baselines"+str(sim), baselines)


np.seterr(over = 'warn', under = 'warn')

d = 10
n_events = 1000
#####################
### HOMOGENOUS ######
#####################

### make this a function####
homogenous = True

timestamps = hom_poisson(d, n_events)

models = ["HkEM", "ADM4", "SG", "Cumulant", "Granger", "ExpKern", "NetInf"]

results_sqe = [[] for i in models]
results_musqe = [[] for i in models]

model_em = pp_models.HkEM()
model_em.optimize_params(timestamps, n_iter=50)

model_adm4 = pp_models.ADM4()
model_adm4.optimize_params(timestamps, n_iter=100)

model_expkern = pp_models.ExpKern()
model_expkern.optimize_params(timestamps, n_iter=100)

model_hc = pp_models.HC()
model_hc.optimize_params(timestamps, n_iter=100)

model_gb = pp_models.Wold()
model_gb.optimize_params(timestamps, n_iter=100)

num_sims = 200
for sim in range(num_sims):
    print(sim)

    timestamps = hom_poisson(d, n_events)
    
    model_em.fit(timestamps)
    Alpha = model_em.get_kernel_norms()
    mu = model_em.get_baselines()
    results_sqe[0].append(square_error(Alpha))
    results_musqe[0].append(mu_se(mu))
    save_model(model_em.name, Alpha, mu, homogenous, sim)
    
    model_adm4.fit(timestamps)
    Alpha = model_adm4.get_kernel_norms()
    mu = model_adm4.get_baselines()
    results_sqe[1].append(square_error(Alpha))
    results_musqe[1].append(mu_se(mu))
    save_model(model_adm4.name, Alpha, mu, homogenous, sim)

    model_sg = pp_models.SG()
    model_sg.fit(timestamps)
    Alpha = model_sg.get_kernel_norms()
    mu = model_sg.get_baselines()
    results_sqe[2].append(square_error(Alpha))
    results_musqe[2].append(mu_se(mu))
    save_model(model_sg.name, Alpha, mu, homogenous, sim)
    
    model_hc.fit(timestamps)
    Alpha = model_hc.get_kernel_norms()
    mu = model_hc.get_baselines()
    results_sqe[3].append(square_error(Alpha))
    results_musqe[3].append(mu_se(mu))
    save_model(model_hc.name, Alpha, mu, homogenous, sim)
    
    model_gb.fit(timestamps)
    Alpha = model_gb.get_kernel_norms()
    mu = model_gb.get_baselines()
    results_sqe[4].append(square_error(Alpha))
    results_musqe[4].append(mu_se(mu))
    save_model(model_gb.name, Alpha, mu, homogenous, sim)
    
    model_expkern.fit(timestamps)
    Alpha = model_expkern.get_kernel_norms()
    mu = model_expkern.get_baselines()
    results_sqe[5].append(square_error(Alpha))
    results_musqe[5].append(mu_se(mu))
    save_model(model_expkern.name, Alpha, mu, homogenous, sim)

results_sqe = [sorted(r) for r in results_sqe]
results_musqe = [sorted(r) for r in results_musqe]

alpha = 0.05

f = open("Poisson_result","w")
for model_i in range(len(models)):
    model = models[model_i]
    if len(results_sqe[model_i])>0:
        f.write(model+" Alpha squared error %f %f %f\n"%(results_sqe[model_i][int(num_sims*alpha/2)-1],\
            results_sqe[model_i][int(num_sims/2)],results_sqe[model_i][int(num_sims*(1-alpha/2))]))
        f.write(model+" mu squared error %f %f %f\n"%(results_musqe[model_i][int(num_sims*alpha/2)-1],\
            results_musqe[model_i][int(num_sims/2)],results_musqe[model_i][int(num_sims*(1-alpha/2))]))

f = open("Poisson_result2","w")
print("END HOMOGENOUS")
d = 10
n_events = 1000
#####################
### INHOMOGENOUS ####
#####################
homogenous = False

results_sqe = [[] for i in models]
results_musqe = [[] for i in models]

timestamps = inhom_poisson(d, n_events, equal=False)

model_em = pp_models.HkEM()
model_em.optimize_params(timestamps, n_iter=30)

model_adm4 = pp_models.ADM4()
model_adm4.optimize_params(timestamps, n_iter=200)

model_expkern = pp_models.ExpKern()
model_expkern.optimize_params(timestamps, n_iter=100)

model_hc = pp_models.HC()
model_hc.optimize_params(timestamps, n_iter=100)

model_gb = pp_models.Wold()
model_gb.optimize_params(timestamps, n_iter=100)


num_sims = 200
for sim in range(num_sims):
    print(sim)

    timestamps = inhom_poisson(d, n_events, equal=False)
    
    model_em.fit(timestamps)
    Alpha = model_em.get_kernel_norms()
    mu = model_em.get_baselines()
    results_sqe[0].append(square_error(Alpha))
    results_musqe[0].append(mu_se(mu))
    save_model(model_em.name, Alpha, mu, homogenous, sim)
    
    model_adm4.fit(timestamps)
    Alpha = model_adm4.get_kernel_norms()
    mu = model_adm4.get_baselines()
    results_sqe[1].append(square_error(Alpha))
    results_musqe[1].append(mu_se(mu))
    save_model(model_adm4.name, Alpha, mu, homogenous, sim)
    
    model_sg = pp_models.SG()
    model_sg.fit(timestamps)
    Alpha = model_sg.get_kernel_norms()
    mu = model_sg.get_baselines()
    results_sqe[2].append(square_error(Alpha))
    results_musqe[2].append(mu_se(mu))
    save_model(model_sg.name, Alpha, mu, homogenous, sim)
    
    model_hc.fit(timestamps)
    Alpha = model_hc.get_kernel_norms()
    mu = model_hc.get_baselines()
    results_sqe[3].append(square_error(Alpha))
    results_musqe[3].append(mu_se(mu))
    save_model(model_hc.name, Alpha, mu, homogenous, sim)
    
    model_gb.fit(timestamps)
    Alpha = model_gb.get_kernel_norms()
    mu = model_gb.get_baselines()
    results_sqe[4].append(square_error(Alpha))
    results_musqe[4].append(mu_se(mu))
    save_model(model_gb.name, Alpha, mu, homogenous, sim)
    
    model_expkern.fit(timestamps)
    Alpha = model_expkern.get_kernel_norms()
    mu = model_expkern.get_baselines()
    results_sqe[5].append(square_error(Alpha))
    results_musqe[5].append(mu_se(mu))
    save_model(model_expkern.name, Alpha, mu, homogenous, sim)
    
results_sqe = [sorted(r) for r in results_sqe]
results_musqe = [sorted(r) for r in results_musqe]
    
alpha = 0.05

for model_i in range(len(models)):
    model = models[model_i]
    if len(results_sqe[model_i])>0:
        f.write(model+" Alpha squared error %f %f %f\n"%(results_sqe[model_i][int(num_sims*alpha/2)-1],\
                results_sqe[model_i][int(num_sims/2)],results_sqe[model_i][int(num_sims*(1-alpha/2))]))
        f.write(model+" mu squared error %f %f %f\n"%(results_musqe[model_i][int(num_sims*alpha/2)-1],\
                results_musqe[model_i][int(num_sims/2)],results_musqe[model_i][int(num_sims*(1-alpha/2))]))

f.close()