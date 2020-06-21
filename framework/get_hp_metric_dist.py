import numpy as np
import sklearn.preprocessing
import os
import pp_models
from time import time
from aux_funcs import *
from sklearn.model_selection import ParameterSampler

def write_results_time(filepath, model, run_time, timestamps):
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    with open(filepath+"/parameters.txt", "w") as fparams:
        fparams.write(str(model.get_params()))
    with open(filepath+"/time.txt", "w") as ftime:
        ftime.write(str(run_time))
    with open(filepath+"/score.txt", "w") as fscore:
        fscore.write(str(model.score(timestamps)))
    np.save(filepath+"/adjacency", model.get_kernel_norms())
    np.save(filepath+"/baselines", model.get_baselines())
    np.save(filepath+"/baselines", model.get_baselines())



def test_model(model, timestamps, filepath):
    tic = time()
    model.fit(timestamps)
    run_time = time() - tic
    write_results_time(filepath, model, run_time, timestamps)

file_list = ["sx-mathoverflow", "sx-askubuntu", "sx-superuser", "CollegeMsg", "email-Eu-core-temporal", "wiki-talk-temporal","memetracker_2009-01"]
file_list = ["memetracker_2009-01"]

datapath = {}
for file in file_list:
    #datapath[file] = (address_to_file_with_timestamps, address to groundtruth matrix)


n_iter = 100
np.seterr(over = 'warn', under = 'warn')
for file in file_list:
    timestamps, groundTruth = read_timestamps_groundtruth(datapath[file][0],datapath[file][1])
    print(file)
    
    
    model_bk = pp_models.BasisKernels()
    i = 0
    for param_set in ParameterSampler(model_bk.param_dist, n_iter=n_iter):
        i += 1 
        try:
            model_bk.set_params(param_set["kernel_support"], param_set["n_basis"], param_set["kernel_size"], param_set["C"], param_set["tol"])
            test_model(model_bk, timestamps, "results/test_hp/"+file+"/"+model_bk.name+"/run"+str(i))
        except:
            print(i, "ERROR BK")
    
    model_cl = pp_models.CondLaw()
    test_model(model_cl, timestamps, "results/"+file+"/"+model_cl.name)
    
    
    model_hc = pp_models.HC()
    i = 0
    for param_set in ParameterSampler(model_hc.param_dist, n_iter=n_iter):
        i += 1 
        try:
            model_hc.set_params(param_set["integration_support"], param_set["C"], param_set["penalty"], param_set["solver"], param_set["step"], param_set["tol"])
            test_model(model_hc, timestamps, "results/test_hp/"+file+"/"+model_hc.name+"/run"+str(i))
        except:
            print(i, "ERROR HC")
    

    model_adm4 = pp_models.ADM4()
    i = 0
    for param_set in ParameterSampler(model_adm4.param_dist, n_iter=n_iter):
        i += 1 
        model_adm4.set_params(param_set["decay"],param_set["C"],param_set["lasso_nuclear_ratio"],param_set["tol"])
        test_model(model_adm4, timestamps, "results/test_hp/"+file+"/"+model_adm4.name+"/run"+str(i))
    
    model_gb = pp_models.Wold()
    i = 0
    for param_set in ParameterSampler(model_gb.param_dist, n_iter=n_iter):
        i += 1 
        model_gb.set_params(param_set["beta"])
        test_model(model_gb, timestamps, "results/test_hp/"+file+"/"+model_gb.name+"/run"+str(i))
    
    model_em = pp_models.HkEM()
    i = 0
    list_iters = list(range(1,53)) + list(range(77,101))
    j = 0
    for param_set in ParameterSampler(model_em.param_dist, n_iter=len(list_iters)):
        i = list_iters[j] 
        j += 1
        model_em.set_params(param_set['kernel_support'], param_set['kernel_size'],param_set['tol'])
        test_model(model_em, timestamps, "results/test_hp/"+file+"/"+model_em.name+"/run"+str(i))
    
    model_ek = pp_models.ExpKern()
    i = 0
    for param_set in ParameterSampler(model_ek.param_dist, n_iter=n_iter):
        i += 1
        try:
            model_ek.set_params(param_set["decays"], param_set["C"], param_set["penalty"], param_set["solver_exp"], param_set["step"], param_set["tol"])
            test_model(model_ek, timestamps, "results/test_hp/"+file+"/"+model_ek.name+"/run"+str(i))
        except:
            print(i, "ERROR ExpKern")

    model_sg = pp_models.SG()
    i = 0
    for param_set in ParameterSampler(model_sg.param_dist, n_iter=n_iter):
        i += 1 
        model_sg.set_params(param_set["max_mean_gaussian"], param_set["n_gaussians"], param_set["step_size"], param_set["C"], param_set["lasso_grouplasso_ratio"], param_set["tol"])
        test_model(model_sg, timestamps, "results/test_hp/"+file+"/"+model_sg.name+"/run"+str(i))
    
    
    model_netinf = pp_models.netinf()
    i = 0
    for param_set in ParameterSampler(model_netinf.param_dist, n_iter=n_iter):
        i += 1 
        model_netinf.set_params(param_set["alpha"], param_set["dist"])
        test_model(model_netinf, timestamps, "results/test_hp/"+file+"/"+model_netinf.name+"/run"+str(i))
    