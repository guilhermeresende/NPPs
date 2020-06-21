import numpy as np
import scipy.stats as ss
import scipy.sparse.csgraph
import sklearn
import sklearn.metrics
import sys
sys.path.append("../granger-busca")
from gb import simulate, GrangerBusca
import tick.hawkes as hk
from bisect import bisect


def divide_training_test(timestamps, split_time):
    d = len(timestamps)
    training_set = [[] for i in range(d)]
    test_set = [[] for i in range(d)]

    for p1 in range(len(timestamps)):
        for t in timestamps[p1]:
            if(t<split_time):
                training_set[p1].append(t)
            else:
                test_set[p1].append(t)

    training_set = [np.array(i) for i in training_set]
    test_set = [np.array(i) for i in test_set]
    return(training_set, test_set)

def divide_training_test_prop(timestamps, proportion = 0.5):
    d = len(timestamps)
    total_points = sum([len(i) for i in timestamps])

    finish_time = 0
    pct_points_test = 0
    list_pointers = [0]*d
    next_update = list(range(d))
    next_update = sorted(next_update, key = lambda x:timestamps[x][0])

    training_set = [[] for i in range(d)]
    test_set = timestamps[:]
    while(pct_points_test<proportion):
        next_inserted = next_update.pop(0)

        removed = test_set[next_inserted][0]
        test_set[next_inserted] = np.delete(test_set[next_inserted], 0)

        training_set[next_inserted].append(removed)

        i = 0
        while(i<len(next_update) and test_set[next_inserted][0]>test_set[next_update[i]][0]):
            i += 1
        next_update.insert(i, next_inserted)
        pct_points_test += 1./total_points
    return (training_set, test_set)

def split_filter_train_test(timestamps, groundTruth, percentTest, gtMu=None, min_events=30):

    minT = min([j for i in timestamps for j in i])
    maxT = max([j for i in timestamps for j in i])
    simTime = maxT-minT

    (train, test) = divide_training_test(timestamps, split_time = ((1-percentTest)*maxT + percentTest*minT))
    newTrain = []
    newTest = []
    for p1 in range(len(train)):
        if len(train[p1]) >= min_events and len(test[p1]) >= min_events:
            newTrain.append(train[p1])
            newTest.append(test[p1])
        else:
            groundTruth=np.delete(groundTruth,len(newTrain),axis=0)
            groundTruth=np.delete(groundTruth,len(newTrain),axis=1)
            if gtMu is not None:
                gtMu = np.delete(gtMu,len(newTrain))

    train = newTrain
    test = newTest
    if gtMu is not None:
        return (train,test,groundTruth,gtMu)
        
    return (train,test,groundTruth)


'''    
def precision(A_true, A_pred, k = 10):
    res = 0.0
    precisionl = []
    tmp = 0
    for i in range(A_true.shape[0]):
        x = A_true[i]
        x = x[x != 0]
        #print(i)
        #print(A_true[i])
        y = A_pred[i]
        y = y[y != 0]
        #print(A_pred[i])
        kx = min(len(x), k)
        ky = min(len(y), k)
        if ky == 0 or kx == 0: continue
        x = set(np.argpartition(x, -kx)[-kx:])
        y = set(np.argpartition(y, -ky)[-ky:])
        res += len(x.intersection(y)) / k
        #print(i, res)
        precisionl.append(len(x.intersection(y)) / k)
        tmp += 1
    if tmp == 0: return 0
    return res / tmp
'''


def rankcorr(A_true, A_pred, pvalues = False):

    assert len(A_true) == len(A_pred)
    if not(np.any(A_pred)):
        return 0

    ts = []
    for i in range(len(A_pred)):
        tau = ss.kendalltau(A_true[i], A_pred[i])
        t = tau[0]
        pvalue = tau[1]
        if not np.isnan(t):
            if (pvalues):
                ts.append(pvalue)
            else:
                ts.append(t)
    if (pvalues):
        return ts
    return np.mean(ts)

def precision(A_true, A_pred, k = 10):
    res = 0.0
    precisionl = []
    for i in range(A_true.shape[0]):
        k_not_0 = np.sum(A_true[0]>0)
        sorted_idx_pred = sorted(range(len(A_pred[i])), key=lambda x:A_pred[i][x], reverse=True)[:k_not_0]
        sorted_idx_true = sorted(range(len(A_true[i])), key=lambda x:A_true[i][x], reverse=True)[:k_not_0]
        precisionl.append(len(np.intersect1d(sorted_idx_pred,sorted_idx_true,assume_unique=True))/k_not_0) 
    return np.mean(precisionl)

def rel_err(A_true, A_pred):
    err = []
    for i in range(len(A_true)):
        denom = np.abs(np.where(A_true[i]==0,1,A_true[i]))
        v = np.sum(np.abs(A_true[i]-A_pred[i])/denom)
        err.append(v)
    return np.mean(err)

def rel_err2(A_true, A_pred):
    err = []
    for i in range(len(A_true)):
        v=0.0
        for j in range(len(A_true[i])):
            if A_true[i][j]!=0:
                v+=abs(A_true[i][j]-A_pred[i][j])/abs(A_true[i][j])
            else:
                v+=abs(A_pred[i][j])
        err.append(v)
    return np.mean(err)

def rel_err_baseline(ground_truth, baselines):
    rel_err = 0
    err = []
    for i in range(len(baselines)):
        err.append(baselines[i]-ground_truth[i])**2
        rel_err+=(baselines[i]-ground_truth[i])**2
    rel_err= rel_err**0.5 / (sum([mu**2 for mu in ground_truth]))**0.5
    return np.mean(rel_err)

def ndcg_bin(A_true, A_pred):
    dcgp_avg = []
    for i in range(len(A_pred)):
        dcgp = 0
        idcgp = 0
        sorted_idx = sorted(range(len(A_pred[i])), key=lambda x:A_pred[i][x], reverse=True)
        gt_idx = 0
        for j in range(len(A_pred[i])):
            if A_true[i][sorted_idx[j]] != 0:
                dcgp += (1)/np.log2(j+2)
            if A_true[i][j] != 0:
                idcgp += (1)/np.log2(gt_idx+2)
                gt_idx +=1
        if(np.any(A_true[i])):
            dcgp_avg.append(dcgp/idcgp)
    assert len(dcgp_avg) > 0
    return np.mean(dcgp_avg)

def ndcg2(A_true, A_pred):
    dcgp_avg = []
    for i in range(len(A_pred)):
        dcgp = 0
        idcgp = 0
        sorted_idx_true = sorted(range(len(A_true[i])), key=lambda x:A_true[i][x], reverse=True)
        sorted_idx_pred = sorted(range(len(A_pred[i])), key=lambda x:A_pred[i][x], reverse=True)
        gt_idx = 0
        for j in range(len(A_pred[i])):
            relpred = A_true[i][sorted_idx_pred[j]]
            reltrue = A_true[i][sorted_idx_true[j]]
            if A_pred[i][sorted_idx_pred[j]] > 0:
                dcgp += (2**relpred - 1)/np.log2(j+2)
            idcgp += (2**reltrue - 1)/np.log2(j+2)
        if(np.any(A_true[i])):
            dcgp_avg.append(dcgp/idcgp)
    assert len(dcgp_avg) > 0
    return np.mean(dcgp_avg)

def ndcg(A_true, A_pred):
    dcgp_avg = []
    for i in range(len(A_pred)):
        dcgp = 0
        idcgp = 0
        sorted_idx_true = sorted(range(len(A_true[i])), key=lambda x:A_true[i][x], reverse=True)
        sorted_idx_pred = sorted(range(len(A_pred[i])), key=lambda x:A_pred[i][x], reverse=True)
       
        p_sorted_by_pred = A_pred[i][sorted_idx_pred]
        reltrue = A_true[i][sorted_idx_true]
        #reltrue = reltrue[p_sorted_by_pred != 0]
        relpred = np.where(p_sorted_by_pred==0, 0, A_true[i][sorted_idx_pred])
        #relpred = relpred[p_sorted_by_pred != 0]
  
        n = len(A_pred[i])
        dcgp = np.sum((2**relpred - 1)/np.log2(np.arange(n)+2))
        idcgp = np.sum((2**reltrue - 1)/np.log2(np.arange(n)+2))
        all_zeros_true = np.all(A_true[i] == 0)
        all_zeros_pred = np.all(A_pred[i] == 0)
        if all_zeros_pred and all_zeros_true:
            dcgp_avg.append(1.)
        elif all_zeros_pred and not all_zeros_true:
            dcgp_avg.append(0.)
        elif not all_zeros_pred and not all_zeros_true:
            dcgp_avg.append(dcgp/idcgp)
            
    assert len(dcgp_avg) > 0
    return np.mean(dcgp_avg)


def roc_auc(A_true, A_pred):
    binGT = A_true > 0
    true_values = binGT.flatten()
    y_values = A_pred.flatten()
    rocl = []
    for i in range(len(A_pred)):
        if np.any(binGT[i]):
            rocl.append(sklearn.metrics.roc_auc_score(binGT[i], A_pred[i]))
    return np.mean(rocl)
    #return sklearn.metrics.roc_auc_score(true_values, y_values)

def nrmse2(A_true, A_pred):
    err = []
    for i in range(len(A_true)):
        v = 0
        for j in range(len(A_true[i])):
            v += (A_true[i][j]-A_pred[i][j])**2
        if(np.any(A_true[i])):
            v = ((1/len(A_true[i]))*v)**0.5/(max(A_true[i])-min(A_true[i]))
            err.append(v)
    assert len(err) > 0
    return np.mean(err)

def nrmse(A_true, A_pred):
    err = []
    for i in range(len(A_true)):
        v = np.sum((A_true[i]-A_pred[i])**2)
        if(np.any(A_true[i])):
            v = ((1/len(A_true[i]))*v)**0.5/(max(A_true[i])-min(A_true[i]))
            err.append(v)
    assert len(err) > 0
    return np.mean(err)

def apbin(A_true, A_pred):
    map_avg = []
    binGT = A_true > 0.
    for i in range(len(A_pred)):
        ap = 0
        sorted_idx = sorted(range(len(A_pred[i])), key=lambda x:A_pred[i][x], reverse=True)
        num_correct = 0
        for k in range(1,len(A_pred[i])+1):
            if (binGT[i][sorted_idx[k-1]]) and (A_pred[i][sorted_idx[k-1]]):
                num_correct += 1
                ap += num_correct/k
        if np.sum(binGT[i]) > 0 :
            map_avg.append(ap/np.sum(binGT[i]))  
    return np.mean(map_avg)

def apk(A_true, A_pred):
    map_avg = []
    for i in range(len(A_pred)):
        idx_pred = np.arange(len(A_pred[i]))[A_pred[i]!=0]
        sorted_idx_pred = np.array(sorted(idx_pred, key=lambda x:A_pred[i][x], reverse=True))
        sorted_idx_true = np.array(sorted(range(len(A_true[i])), key=lambda x:A_true[i][x], reverse=True))
        pk = 0
        numTrue = np.sum(A_true[i]>0)
        for k in range(1,numTrue+1):
            pk += len(np.intersect1d(sorted_idx_pred[:k],sorted_idx_true[:k],assume_unique=True))/k  
        if numTrue > 0:
            map_avg.append(pk/numTrue)  
    return np.mean(map_avg)

def spectrum_laplacian_ksdist(A_true, A_pred, low=True):
    sym_A_true = np.copy(A_true)
    sym_A_pred = np.copy(A_pred)

    i_triang = np.tril_indices(A_true.shape[0], 0)
    if low:
        sym_A_true.T[i_triang] = sym_A_true[i_triang]
        sym_A_pred.T[i_triang] = sym_A_pred[i_triang]
    else:
        sym_A_true[i_triang] = sym_A_true.T[i_triang]
        sym_A_pred[i_triang] = sym_A_pred.T[i_triang]
    L_true = scipy.sparse.csgraph.laplacian(sym_A_true)
    L_pred = scipy.sparse.csgraph.laplacian(sym_A_pred)

    eigen_true = np.linalg.eig(L_true)[0]
    eigen_pred = np.linalg.eig(L_pred)[0]
    
    return ss.ks_2samp(eigen_true, eigen_pred)[0]

def det_diff(A_true, A_pred):
    diff = A_true - A_pred
    return np.linalg.det(diff)

def generalized_distance(A_true, A_pred, swap_cost = lambda pos:1, dist_metric = lambda x,y: 1):
    distlist = []
    for i in range(len(A_pred)):
        v1 = A_true[i]
        v2 = A_pred[i]
        rank1 = [int(i) for i in ss.rankdata(v1)]
        rank2 = [int(i) for i in ss.rankdata(v2)]
        dist = 0
        for idx1 in range(len(v1)):
            for idx2 in range(idx1, len(v1)):
                if (v1[idx1]<v1[idx2] and v2[idx1]>v2[idx2]) or (v1[idx1]>v1[idx2] and v2[idx1]<v2[idx2]):
                    pi1 = sum([swap_cost(i) for i in range(1,rank1[idx1])])
                    pi2 = sum([swap_cost(i) for i in range(1,rank1[idx2])])
                    pi_bar = (pi1-pi2)/(rank1[idx1]-rank1[idx2])

                    pj1 = sum([swap_cost(i) for i in range(1,rank2[idx1])])
                    pj2 = sum([swap_cost(i) for i in range(1,rank2[idx2])])
                    pj_bar = (pj1-pj2)/(rank2[idx1]-rank2[idx2])
                    dist += v1[idx1]*v2[idx2]*pi_bar*pj_bar*dist_metric(rank2[idx1],rank2[idx2])
        distlist.append(dist)
    return np.mean(distlist)

def simulate_method_result(model, num_simulations = 10, simulation_time = 100, kernel_support = 5):    
    mu = model.baseline
    list_simulations = []
    kernels = []
    ext=0
    for i in range(model.n_nodes):
        kernels.append([])
        for j in range(model.n_nodes):

            t_values = np.linspace(0, kernel_support-1/20., kernel_support*20)
            y_values = model.get_kernel_values(i, j,t_values)
            begin_t=kernel_support
            while(y_values[-1]>1e-5): #adds time until kernel values are close to 0
                t_values_next = np.linspace(begin_t, begin_t+kernel_support-1/20., kernel_support*20)
                y_values_next = model.get_kernel_values(i, j,t_values_next)
                t_values = np.concatenate((t_values, t_values_next))
                y_values = np.concatenate((y_values, y_values_next))
                begin_t = begin_t+kernel_support
                ext+=1.0
            
            kernels[i].append(hk.HawkesKernelTimeFunc(t_values = t_values, y_values = y_values))
    print("average exts:",ext/(model.n_nodes**2))
    for i in range(num_simulations):
        #print(i)
        #h = hk.SimuHawkes(kernels = kernels, baseline = (mu), end_time =simulation_time, force_simulation = True)
        h = hk.SimuHawkes(kernels = kernels, baseline = (mu), end_time =simulation_time, verbose = False)
        h.threshold_negative_intensity(True)
        h.simulate()
        list_simulations.append(h.timestamps)
    return list_simulations

def simulate_method_result_granger(model, num_simulations = 10, simulation_time = 100):    
    mu = model.mu_
    Alpha = sklearn.preprocessing.normalize(model.Alpha_.toarray(), "l1")
    Beta = np.vstack((model.beta_,)*len(mu))
    list_simulations = []
    for i in range(num_simulations):
        t1 = time()
        sim = simulate.GrangeBuscaSimulator(mu_rates = mu, Alpha_ba = Alpha, Beta_ba = np.vstack((model.beta_,)*len(mu)))
        list_simulations.append(sim.simulate(simulation_time))
    return list_simulations


def MSEN(simulations, ground_truth):
    MSEN = 0.
    msenl = []
    for i in range(len(simulations)):
        for process in range(len(ground_truth)):
            MSEN += (len(simulations[i][process]) - len(ground_truth[process]))**2
            msenl.append((len(simulations[i][process]) - len(ground_truth[process]))**2)
    return msenl

def MEN(simulations, ground_truth):
    MEN = 0.
    menl = []
    for i in range(len(simulations)):
        for process in range(len(ground_truth)):
            MEN += len(simulations[i][process]) - len(ground_truth[process])
            menl.append(len(simulations[i][process]) - len(ground_truth[process]))
    return menl

def fuzzy_match(simulations, ground_truth, tolerance = 100):
    match = 0.
    matchl = []
    for i in range(len(simulations)):
        for process in range(len(ground_truth)):
            if abs(len(simulations[i][process]) - len(ground_truth[process])) <= tolerance:
                match += 1.
                matchl.append(1)
            else:
                matchl.append(0)
    return matchl

def MAPEn(simulations, ground_truth):
    MAPE = 0.
    mapel = []
    for i in range(len(simulations)):
        for process in range(len(ground_truth)):
            if(len(ground_truth[process])>0):
                mapel.append(abs((len(simulations[i][process]) - len(ground_truth[process]))/len(ground_truth[process])))
                MAPE += abs((len(simulations[i][process]) - len(ground_truth[process]))/len(ground_truth[process]))
                mapel.append(abs((len(simulations[i][process]) - len(ground_truth[process]))/len(ground_truth[process])))
            else:
                mapel.append(abs((len(simulations[i][process]) - len(ground_truth[process]))))
                MAPE += abs((len(simulations[i][process]) - len(ground_truth[process])))
                mapel.append(abs((len(simulations[i][process]) - len(ground_truth[process]))))
    return mapel


def get_deltas(timestamps):
    deltas = [[] for i in range(len(timestamps))]
    for p1 in range(len(timestamps)):
        for t_idx in range(1, len(timestamps[p1])):
            deltas[p1].append(timestamps[p1][t_idx] - timestamps[p1][t_idx-1])
    return [np.array(delta) for delta in deltas]

def diff_ks(simulations, ground_truth):
    ks_sum = 0
    ksl = []
    cnt_valid = 0.
    for simulated in simulations:
        deltas = get_deltas(simulated)
        deltas_gt = get_deltas(ground_truth)
        for p1 in range(len(deltas)):
            if (len(deltas[p1])>1 and len(deltas_gt[p1])>1):
                ks_sum += ss.ks_2samp(deltas[p1], deltas_gt[p1])[0]
                ksl.append(ss.ks_2samp(deltas[p1], deltas_gt[p1])[0])
                cnt_valid += 1.
    return ksl
        
def diff_corr(simulations, ground_truth):
    corr_sum = 0
    corrl = []
    cnt_valid = 0.
    for simulated in simulations:
        deltas = get_deltas(simulated)
        deltas_gt = get_deltas(ground_truth)
        for p1 in range(len(deltas)):
            if (len(deltas[p1])>3 and len(deltas_gt[p1])>3):
                deltas_minus1 = deltas[p1][:-1]
                deltas_p = deltas[p1][1:] 

                corr = ss.pearsonr(deltas_p, deltas_minus1)[0]
                
                deltas_minus1_gt = deltas_gt[p1][:-1]
                deltas_p_gt = deltas_gt[p1][1:]            
                corr_gt = ss.pearsonr(deltas_p_gt, deltas_minus1_gt)[0]
                
                corr_sum += abs(corr-corr_gt)
                corrl.append(abs(corr-corr_gt))
                cnt_valid += 1.
           
    return corrl

def diff_var(simulations, ground_truth):
    var_diff = 0
    varl = []
    cnt_valid = 0.
    for simulated in simulations:
        deltas = get_deltas(simulated)
        deltas_gt = get_deltas(ground_truth)
        for p1 in range(len(deltas)):
            if (len(deltas[p1])>1 and len(deltas_gt[p1])>1):
                var_ = np.var(deltas[p1])
                var_gt = np.var(deltas_gt[p1])
                var_diff += abs(var_-var_gt)
                varl.append(abs(var_-var_gt))
                cnt_valid += 1.
    return varl

def diff_coef_var(simulations, ground_truth):
    coef_diff = 0
    coefl = []
    cnt_valid = 0.
    for simulated in simulations:
        deltas = get_deltas(simulated)
        deltas_gt = get_deltas(ground_truth)
        for p1 in range(len(deltas)):
            if (len(deltas[p1])>1 and len(deltas_gt[p1])>1):
                coef_ = np.std(deltas[p1])/np.mean(deltas[p1])
                coef_gt = np.std(deltas_gt[p1])/np.mean(deltas_gt[p1])
                
                coef_diff += abs(coef_-coef_gt)
                coefl.append(abs(coef_-coef_gt))
                cnt_valid += 1.
    return coefl

def round_to_base(num, base):
    return base*round(num/base)


def likelihood_calc(model, timestamps):
    delta_t = 0.1
    T = max([j for i in timestamps for j in i])
    T = round_to_base(T, delta_t) 
    intensity = [np.linspace(model.baseline[i], model.baseline[i], T/delta_t+1) for i in range(len(timestamps))]
    #intensity = [np.linspace(0.01, 0.01, T/delta_t+1) for i in range(len(timestamps))]
    log_lik = 0

    for p1 in range(len(timestamps)):
        for t in timestamps[p1]:      
            t_rounded = round_to_base(t, delta_t) 
            
            for p2 in range(len(timestamps)):   
                dif = round(round_to_base((T-t_rounded)/delta_t, delta_t))   
                t_values = np.linspace(0, round_to_base(T-t_rounded, delta_t), dif+1) 
                kernel_values = model.get_kernel_values(p2, p1, t_values)
                #kernel_values = kernels[p2][p1].get_values(t_values)   
                idx = round((round_to_base(t, delta_t)/delta_t))
                assert idx == len(intensity[p2])-len(kernel_values)
                kernel_values = np.concatenate((np.linspace(0, 0,len(intensity[p2])-len(kernel_values)), kernel_values))                
                intensity[p2] = intensity[p2]+kernel_values
          
    log_lik = 0
    for p1 in range(len(timestamps)):
        
        k_idx = 0
        for t in timestamps[p1]:            
            idx = int(round((round_to_base(t, delta_t)/delta_t)))-1
            log_lik += np.log(intensity[p1][idx])
         
        for k_idx in range(len(intensity[p1])-1):
            if(intensity[p1][k_idx]>1e-300):
                log_lik -= delta_t*intensity[p1][k_idx]

    return log_lik

def calc_delta(p1, p2, t_idx, timestamps):

    tp = timestamps[p1][t_idx]
    tpp_idx = bisect(timestamps[p2], tp)
    if tpp_idx == len(timestamps[p2]):
        tpp_idx -= 1
    tpp = timestamps[p2][tpp_idx]
    while tpp >= tp and tpp_idx > 0:
        tpp_idx -= 1
        tpp = timestamps[p2][tpp_idx]
    if tpp >= tp:
        return 0

    return tp - tpp


def granger_loglik(timestamps, model): 
    mu = model.mu_
    Alpha = sklearn.preprocessing.normalize(model.Alpha_.toarray(), "l1")
    Beta = np.vstack((model.beta_,)*len(mu))
    assert len(timestamps) == len(mu)
    T = max([j for i in timestamps for j in i])
    p = 0
    d = len(timestamps)
    for p1 in range(d):
        past_term = [0]*d
        for t_idx in range(len(timestamps[p1])):
            delta_ba = 0
            first_term = 0
            integral = 0
            for p2 in range(d):
                delta_ba = calc_delta(p1,p2,t_idx,timestamps)              
                if(delta_ba>0):
                    first_term += Alpha[p2][p1] / (Beta[p2][p1]+delta_ba)
                integral += past_term[p2] * (timestamps[p1][t_idx]-timestamps[p1][t_idx-1])
                
                if(delta_ba>0):
                    past_term[p2] = Alpha[p2][p1] / (Beta[p2][p1]+delta_ba)                  
                else:
                    past_term[p2] = 0.
            if(mu[p1]+first_term != 0):               
                p += np.log(mu[p1]+first_term)-integral

        end_integral = 0    
        for p2 in range(d):
            if(t_idx>0):
                end_integral += past_term[p2]*(T-timestamps[p1][t_idx])

        p -= end_integral

        p -= T*mu[p1]

    return p