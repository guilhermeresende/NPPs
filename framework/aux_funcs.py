import numpy as np
import sklearn
import os

def read_timestamps_groundtruth(tickspath, groundtruthpath):
    groundTruth = sklearn.preprocessing.normalize(np.array(np.load(groundtruthpath)),'l1')    

    timestamps = []
    with open(tickspath) as data:
        for l in data:    
            timestamps.append(np.array(sorted([float(x) for x in l.split()[1:]])))
    minT = min([t[0] for t in timestamps])
    for i in range(len(timestamps)):
        timestamps[i] = np.array([t - minT for t in list(dict.fromkeys(timestamps[i]))])

    return (timestamps,groundTruth)


def read_adjacency(filepath):
    return np.load(filepath+"adjacency.npy")

def read_baselines(filepath):
    return np.load(filepath+"baselines.npy")

def read_simulations(filepath):
    filepathSim = filepath +"simulations"
    sims = []
    for simfile in os.listdir(filepathSim):
        sim = []
        with open(filepathSim+"/"+simfile) as f:
            for line in f:
                sim.append([float(t) for t in line.strip("\n").split(" ") if t!=""])
        sims.append(sim)
    return sims

def read_bootstrap(filepath):
    boot_list = []
    filepathSim = filepath 
    for folder in os.listdir(filepath):
        if "bootstrap" in folder:
            adjacency = np.load(filepath+"/"+folder+"/adjacency.npy")
            boot_list.append(adjacency)
    return boot_list


def read_params(file,modelName):
    filepath = "results/"+file+"/"+modelName+"/parameters.txt"
    params = open(filepath,"r").readline()
    return eval(params)


