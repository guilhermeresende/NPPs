import numpy as np
import sklearn.preprocessing
import os
import pp_models
from aux_funcs import *


def write_learner_results(filepath, model, simulations=None):
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    with open(filepath+"/parameters.txt", 'w') as fparams:
        fparams.write(str(model.get_params()))

    np.save(filepath+"/adjacency", model.get_kernel_norms())
    np.save(filepath+"/baselines", model.get_baselines())

    if simulations is None:
        return 1
    if not os.path.exists(filepath+"/simulations"):
        os.makedirs(filepath+"/simulations")
    for i in range(len(simulations)):
        with open(filepath+"/simulations/"+str(i),'w') as fsims:
            for proc_a in simulations[i]:
                for time in proc_a:
                    fsims.write(str(time)+" ")
                fsims.write('\n')

def test_model(model, timestamps, filepath):
    model.fit(timestamps)
    write_learner_results(filepath, model)
    
def bootstrap3(timestamps , num_divs = 3, B = 10):
    d = len(timestamps)
    T = max([p[-1] for p in timestamps])
    minT = min([p[0] for p in timestamps])
    divisions = [[] for i in range(d)]
    for i in range(num_divs):
        offset1 = i*(T-minT)/num_divs + minT
        offset2 = (i+1)*(T-minT)/num_divs + minT
        for p in range(d):
            divisions[p].append(timestamps[p][(timestamps[p]>offset1) & (timestamps[p]<=offset2)])
    boot_series = []
    fail = 0
    while len(boot_series)<B:
        new_t = [[] for i in range(d)]
        for i in range(num_divs):
            div = np.random.randint(num_divs)
            offset = (i - div)*((T-minT)/num_divs)
            for p in range(d):
                new_t[p] += list(divisions[p][div] + offset)
        if np.all([len(timestamp)>0 for timestamp in new_t]):
            boot_series.append([np.array(timestamp) for timestamp in new_t])
        else:
            fail += 1
    print("Bootstrap failed",fail,"times")
    return boot_series

def bootstrap2(timestamps , num_divs = 3, B = 10):
    d = len(timestamps)
    T = max([p[-1] for p in timestamps])
    minT = min([p[0] for p in timestamps])
    divisions = [[] for i in range(d)]
    for i in range(num_divs):
        offset1 = i*(T-minT)/num_divs + minT
        offset2 = (i+1)*(T-minT)/num_divs + minT
        for p in range(d):
            divisions[p].append(timestamps[p][(timestamps[p]>offset1) & (timestamps[p]<=offset2)])
    boot_series =[]
    for b in range(B):
        new_t = [[] for i in range(d)]
        for i in range(num_divs):
            div = np.random.randint(num_divs)
            offset = (i - div)*((T-minT)/num_divs)
            for p in range(d):
                new_t[p] += list(divisions[p][div] + offset)

        boot_series.append([np.array(timestamp) for timestamp in new_t])
    return boot_series

def bootstrap(timestamps , num_divs = 3, B = 10):
    d = len(timestamps)
    T = max([p[-1] for p in timestamps])
    minT = min([p[0] for p in timestamps])
    block_size = (T-minT)/num_divs
    maxblock = (T-block_size)
    boot_series =[]
    for b in range(B):
        new_t = [[] for i in range(d)]
        for i in range(num_divs):
            interval_beg = np.random.uniform(minT, maxblock)
            offset = minT + block_size*i - interval_beg
            interval_end = interval_beg + block_size
            for p in range(d):
                division = timestamps[p][(timestamps[p]>interval_beg) & (timestamps[p]<=interval_end)]
                new_t[p] += list(division + offset)
        assert np.isclose(interval_end+offset, T)
        boot_series.append([np.array(timestamp) for timestamp in new_t])
    return boot_series

#consertar
def test_bootstrap(boot_series, model, file):
    print("bootstrap",end =" ")
    for times_idx in range(len(boot_series)):
        timestamps = boot_series[times_idx]
        
        test_model(model, timestamps, "results/"+file+"/"+model.name+"/bootstrap"+str(times_idx))
   
        print(times_idx,end=" ",flush=True)
    print()

file_list = ["sx-mathoverflow", "sx-askubuntu", "sx-superuser", "CollegeMsg", "email-Eu-core-temporal", "wiki-talk-temporal","memetracker_2009-01"]
file_list = ["memetracker_2009-01"]

datapath = {}
for file in file_list:
    #datapath[file] = (address_to_file_with_timestamps, address to groundtruth matrix)


models_opt = [pp_models.HkEM(),pp_models.ADM4(), pp_models.HC(), pp_models.ExpKern()]
models_not_opt = [pp_models.SG(),pp_models.BasisKernels(), pp_models.CondLaw(), pp_models.Wold()]

np.seterr(over = 'warn', under = 'warn')
for file in file_list:
    timestamps, groundTruth = read_timestamps_groundtruth(datapath[file][0],datapath[file][1])
    print(file)

    boot_series = bootstrap3(timestamps, num_divs=10, B=100)
    assert np.all([np.all([len(t)>0 for t in boot]) for boot in boot_series])
    
    model_netinf = pp_models.netinf()
    model_netinf.optimize_params(timestamps, n_iter=50)
    #model_netinf.read_params("results/"+file+"/"+model.name+"/parameters.txt")
    test_model(model_netinf, timestamps, "results/"+file+"/"+model_netinf.name)
    test_bootstrap(boot_series, model_netinf, file)
    
    
    
    model_em = pp_models.HkEM()
    model_em.optimize_params(timestamps, n_iter=50)
    #model_em.read_params("results/"+file+"/"+model.name+"/parameters.txt")
    test_model(model_em, timestamps, "results/"+file+"/"+model_em.name)
    test_bootstrap(boot_series, model_em, file)

    model_adm4 = pp_models.ADM4()
    model_adm4.optimize_params(timestamps, n_iter=80)
    #model_adm4.read_params("results/"+file+"/"+model.name+"/parameters.txt")
    test_model(model_adm4, timestamps, "results/"+file+"/"+model_adm4.name)
    test_bootstrap(boot_series, model_adm4, file)
    

    
    try:
        model_bk = pp_models.BasisKernels()
        test_model(model_bk, timestamps, "results/"+file+"/"+model_bk.name)
        test_bootstrap(boot_series, model_bk, file)
    except:
        print("ERROR BK")

    model_cl = pp_models.CondLaw()
    test_model(model_cl, timestamps, "results/"+file+"/"+model_cl.name)
    test_bootstrap(boot_series, model_cl, file)
    
    model_hc = pp_models.HC()
    model_hc.optimize_params(timestamps, n_iter=200)
    #model_hc.read_params("results/"+file+"/"+model.name+"/parameters.txt")
    test_model(model_hc, timestamps, "results/"+file+"/"+model_hc.name)
    test_bootstrap(boot_series, model_hc, file)

    model_gb = pp_models.Wold()
    model_gb.optimize_params(timestamps, n_iter=100)
    test_model(model_gb, timestamps, "results/"+file+"/"+model_gb.name)
    test_bootstrap(boot_series, model_gb, file)
    
    model_ek = pp_models.ExpKern()
    model_ek.optimize_params(timestamps, n_iter=80)
    #model_ek.read_params("results/"+file+"/"+model.name+"/parameters.txt")
    test_model(model_ek, timestamps, "results/"+file+"/"+model_ek.name)
    test_bootstrap(boot_series, model_ek, file)
    
    model_sg = pp_models.SG()
    test_model(model_sg, timestamps, "results/"+file+"/"+model_sg.name)
    test_bootstrap(boot_series, model_sg, file)