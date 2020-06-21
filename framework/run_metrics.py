import numpy as np
import tick.hawkes as hk
from pp_metrics import *
import sklearn
import scipy.stats as ss
import os
from random_search import *
from aux_funcs import *


def shuffle_rank(d):
    shuffled = np.array([np.arange(d) for i in range(d)])
    for row in range(d):
        np.random.shuffle(shuffled[row])
    return shuffled

def shuffle_matrix(gt):
    shuffled = np.copy(gt)
    for row in range(gt.shape[0]):
        shuffled[row] = np.random.permutation(shuffled[row])
    return shuffled

def test_null_model(groundTruthMatrix, precisionK, n, f): 
    alpha = 0.05
    d = groundTruthMatrix.shape[0]
    rankCorrlist = []
    p10list = []
    relErrlist = []
    NDCGlist = []
    rocAuclist = []
    NRMSElist = []
    APKlist = []
    APbinlist = []

    for i in range(n):
        shuffled_rank = shuffle_rank(d)
        shuffled_gt = shuffle_matrix(groundTruthMatrix)
        
        rankCorr = rankcorr(groundTruthMatrix, shuffled_rank)
        rankCorrlist.append(rankCorr)
        p10 = precision(groundTruthMatrix, shuffled_rank, k = precisionK)
        p10list.append(p10)
        relErr = rel_err(groundTruthMatrix, shuffled_gt)
        relErrlist.append(relErr)
        NDCG = ndcg(groundTruthMatrix, shuffled_gt)
        NDCGlist.append(NDCG)
        rocAuc = roc_auc(groundTruthMatrix, shuffled_gt)
        rocAuclist.append(rocAuc)
        NRMSE = nrmse(groundTruthMatrix, shuffled_gt)
        NRMSElist.append(NRMSE)
        APK = apk(groundTruthMatrix, shuffled_rank)
        APKlist.append(APK)

    rankCorrlist = sorted(rankCorrlist)
    p10list = sorted(p10list)
    relErrlist = sorted(relErrlist)
    NDCGlist = sorted(NDCGlist)
    rocAuclist = sorted(rocAuclist)
    NRMSElist = sorted(NRMSElist)
    APKlist = sorted(APKlist)

    print("Rank Correlation %f (%f, %f)"%(rankCorrlist[int(n/2)], rankCorrlist[int(n*alpha/2)], rankCorrlist[int(n*(1-alpha/2))-1]))
    f.write("%.3f (%.3f, %.3f)\t"%(rankCorrlist[int(n/2)], rankCorrlist[int(n*alpha/2)], rankCorrlist[int(n*(1-alpha/2))-1]))

    print("Precision@%d %f (%f, %f)"%(precisionK, p10list[int(n/2)], p10list[int(n*alpha/2)], p10list[int(n*(1-alpha/2))-1]))
    f.write("%.3f (%.3f, %.3f)\t"%(p10list[int(n/2)], p10list[int(n*alpha/2)], p10list[int(n*(1-alpha/2))-1]))

    print("Rel Error %f (%f, %f)"%(relErrlist[int(n/2)], relErrlist[int(n*alpha/2)], relErrlist[int(n*(1-alpha/2))-1]))
    f.write("%.3f (%.3f, %.3f)\t"%(relErrlist[int(n/2)], relErrlist[int(n*alpha/2)], relErrlist[int(n*(1-alpha/2))-1]))

    print("ROC AUC %f (%f, %f)"%(rocAuclist[int(n/2)], rocAuclist[int(n*alpha/2)], rocAuclist[int(n*(1-alpha/2))-1]))
    f.write("%.3f (%.3f, %.3f)\t"%(rocAuclist[int(n/2)], rocAuclist[int(n*alpha/2)], rocAuclist[int(n*(1-alpha/2))-1]))

    print("NDCG %f (%f, %f)"%(NDCGlist[int(n/2)], NDCGlist[int(n*alpha/2)], NDCGlist[int(n*(1-alpha/2))-1]))
    f.write("%.3f (%.3f, %.3f)\t"%(NDCGlist[int(n/2)], NDCGlist[int(n*alpha/2)], NDCGlist[int(n*(1-alpha/2))-1]))

    print("NRMSE %f (%f, %f)"%(NRMSElist[int(n/2)], NRMSElist[int(n*alpha/2)], NRMSElist[int(n*(1-alpha/2))-1]))
    f.write("%.3f (%.3f, %.3f)\t"%(NRMSElist[int(n/2)], NRMSElist[int(n*alpha/2)], NRMSElist[int(n*(1-alpha/2))-1]))

    print("APK %f (%f, %f)"%(APKlist[int(n/2)], APKlist[int(n*alpha/2)], APKlist[int(n*(1-alpha/2))-1]))
    f.write("%.3f (%.3f, %.3f)"%(APKlist[int(n/2)], APKlist[int(n*alpha/2)], APKlist[int(n*(1-alpha/2))-1]))

    f.write("\n")

def model_metrics(modelAPred, boot_list, groundTruthMatrix, baseline, f, groundTruthBaselines = None, precisionK=10, hawkes=True):
    if (hawkes):
        modelAPred = modelAPred.T

    rankCorr = rankcorr(groundTruthMatrix, modelAPred)
    conf_interval = confidence_interval(boot_list, groundTruthMatrix, rankcorr, hawkes)
    print("Rank Correlation %f (%f, %f)"%(rankCorr, conf_interval[0],conf_interval[1]))
    f.write("%.3f (%.3f, %.3f)\t"%(rankCorr, conf_interval[0],conf_interval[1]))

    p10 = precision(groundTruthMatrix, modelAPred, k = precisionK)
    conf_interval = confidence_interval(boot_list, groundTruthMatrix, precision, hawkes, precisionK)
    print("Precision@%d %f (%f, %f)"%(precisionK, p10, conf_interval[0],conf_interval[1]))
    f.write("%.3f (%.3f, %.3f)\t"%(p10, conf_interval[0],conf_interval[1]))

    relErr = rel_err(groundTruthMatrix, modelAPred)
    conf_interval = confidence_interval(boot_list, groundTruthMatrix, rel_err, hawkes)
    print("Rel Error %f (%f, %f)"%(relErr, conf_interval[0],conf_interval[1]))
    f.write("%.3f (%.3f, %.3f)\t"%(relErr, conf_interval[0],conf_interval[1]))

    rocAuc = roc_auc(groundTruthMatrix, modelAPred)
    conf_interval = confidence_interval(boot_list, groundTruthMatrix, roc_auc, hawkes)
    print("ROC AUC %f (%f, %f)"%(rocAuc, conf_interval[0],conf_interval[1]))
    f.write("%.3f (%.3f, %.3f)\t"%(rocAuc, conf_interval[0],conf_interval[1]))
    
    NDCG = ndcg(groundTruthMatrix, modelAPred)
    conf_interval = confidence_interval(boot_list, groundTruthMatrix, ndcg, hawkes)
    print("NDCG %f (%f, %f)"%(NDCG, conf_interval[0],conf_interval[1]))
    f.write("%.3f (%.3f, %.3f)\t"%(NDCG, conf_interval[0],conf_interval[1]))

    NRMSE = nrmse(groundTruthMatrix, modelAPred)
    conf_interval = confidence_interval(boot_list, groundTruthMatrix, nrmse, hawkes)
    print("NRMSE %f (%f, %f)"%(NRMSE, conf_interval[0],conf_interval[1]))
    f.write("%.3f (%.3f, %.3f)\t"%(NRMSE, conf_interval[0],conf_interval[1]))

    APK = apk(groundTruthMatrix, modelAPred)
    conf_interval = confidence_interval(boot_list, groundTruthMatrix, apk, hawkes)
    print("APK %f (%f, %f)"%(APK, conf_interval[0],conf_interval[1]))
    f.write("%.3f (%.3f, %.3f)\t"%(APK, conf_interval[0],conf_interval[1]))
    
    if (groundTruthBaselines is not None):
        relErrB = rel_err_baseline(groundTruthBaselines, baseline)
    f.write("\n")

def confidence_interval(boot_list, groundTruthMatrix, metric_func , hawkes=True, precisionK = None):
    results = []
    n = len(boot_list)
    #assert n == 100
    alpha = 0.05 #confidence interval
    for adjacency in boot_list:
        if (hawkes):
            adjacency = adjacency.T
        if precisionK is None:
            results.append(metric_func(groundTruthMatrix,adjacency))
        else:
            results.append(metric_func(groundTruthMatrix,adjacency, k = precisionK))
    results = sorted(results)
    #print(int(n*alpha/2),int(n*(1-alpha/2)))
    return (results[int(n*alpha/2)], results[int(n*(1-alpha/2))-1])


file_list = ["casc50_memetracker","sx-mathoverflow", "sx-askubuntu", "sx-superuser", "CollegeMsg", "email-Eu-core-temporal", "wiki-talk-temporal","memetracker_2009-01"]
file_list = ["sx-mathoverflow", "sx-askubuntu", "sx-superuser", "CollegeMsg", "email-Eu-core-temporal", "wiki-talk-temporal","memetracker_2009-01"]

datapath = {}
for file in file_list:
    #datapath[file] = (address_to_file_with_timestamps, address to groundtruth matrix)

models = ["HkEM", "ADM4", "Cumulant", "ExpKern", "Granger", "NetInf"]
models = ["HkEM", "ADM4", "SG", "Cumulant", "Granger", "ExpKern"]

model_name_dic = {"HkEM":"HkEM", "ADM4":"ADM4", "SG":"MLE-SGLP", "Cumulant":"HC", "Granger":"GB", "ExpKern": "ExpKern", "NetInf": "NetInf"}

np.seterr(over = 'warn', under = 'warn')

f = open("results_www_othermetrics.txt","w")

for file in file_list:
    print(file)
    f.write(file+"\n")
    timestamps, groundTruth = read_timestamps_groundtruth(datapath[file][0],datapath[file][1])

    precisionK = np.ceil(len(groundTruth)/10)

    for model in models:

        print(model_name_dic[model])
        f.write(model_name_dic[model]+"\n") 
        filepath = "results/"+file+"/"+model+"/"

        modelAPred = read_adjacency (filepath)
        #baseline = read_baselines(filepath)
        baseline = None
        boot_list = read_bootstrap(filepath)
        if os.path.exists(filepath +"/simulations"):
            sims = read_simulations(filepath)

        if model == "Granger" :
            model_metrics(modelAPred=modelAPred, boot_list = boot_list, groundTruthMatrix=groundTruth, baseline=baseline, precisionK=precisionK, f=f, hawkes=False)
        else:
            model_metrics(modelAPred=modelAPred, boot_list = boot_list, groundTruthMatrix=groundTruth, baseline=baseline, precisionK=precisionK, f=f)
    f.write("Null model\n")
    print("Null model")
    test_null_model(groundTruthMatrix=groundTruth, precisionK=precisionK, n = 1000, f=f)

