import sys
sys.path.append("../granger-busca")
import numpy as np
from gb import simulate, GrangerBusca
import tick.hawkes as hk
from pp_metrics import *
import scipy.stats as ss
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit

class GrangerExt(BaseEstimator, TransformerMixin):
    def __init__(self, mtype = "ADM4", decay = None, C = None, lasso_nuclear_ratio = None, tol = None, kernel_support = None, kernel_size = None, integration_support = None, penalty = None, solver = None, step = None, max_mean_gaussian = None, n_gaussians = None, beta = None, decays = None, solver_exp = None):
        self.mtype = mtype
        self.error = False
        if(self.mtype == "EM"):
            self.kernel_support = 10. if (kernel_support is None) else kernel_support
            self.kernel_size = 10 if (kernel_size is None) else kernel_size
            self.tol = 1e-05 if (tol is None) else tol
        elif(self.mtype == "ADM4"):
            self.decay = 1. if (decay is None) else decay
            self.C = 1000. if (C is None) else C
            self.lasso_nuclear_ratio = 0.5 if (lasso_nuclear_ratio is None) else lasso_nuclear_ratio
            self.tol = 1e-05 if (tol is None) else tol
        elif(self.mtype == "HC"):
            self.integration_support = 20.0 if (integration_support is None) else integration_support
            self.C = 1000. if (C is None) else C
            self.penalty = 'none' if (penalty is None) else penalty
            self.solver = 'adagrad' if (solver is None) else solver
            self.step = 1 if (step is None) else step
            self.tol = 1e-05 if (tol is None) else tol
        elif(self.mtype == "SG"):
            self.max_mean_gaussian = 20.0 if (max_mean_gaussian is None) else max_mean_gaussian
            self.n_gaussians = 5 if (n_gaussians is None) else n_gaussians
            self.lasso_nuclear_ratio = 0.5 if (lasso_nuclear_ratio is None) else lasso_nuclear_ratio
            self.C = 1000. if (C is None) else C
        elif(self.mtype == "Granger"):
            self.beta = 1 if (beta is None) else beta
        elif(self.mtype == "ExpKern"):
            self.decays = 1 if (decays is None) else decays
            self.C = 1000. if (C is None) else C
            self.penalty = 'none' if (penalty is None) else penalty
            self.solver_exp = 'gd' if (solver_exp is None) else solver_exp
            self.step = 1 if (step is None) else step
            self.tol = 1e-05 if (tol is None) else tol
        elif(self.mtype == "SumExpKern"):
            self.decays = 1 if (decays is None) else decays
            self.C = 1000. if (C is None) else C
            self.penalty = 'none' if (penalty is None) else penalty
            self.solver_exp = 'gd' if (solver_exp is None) else solver_exp
            self.step = 1 if (step is None) else step
            self.tol = 1e-05 if (tol is None) else tol
            
    def fit(self, data):
        timestamps = data[0]
        if(self.mtype == "EM"):
            model = hk.HawkesEM(kernel_support = self.kernel_support, kernel_size = self.kernel_size, tol = self.tol, max_iter = 1000, n_threads=-1)
            model.fit(timestamps)

        elif(self.mtype == "ADM4"):
            model = hk.HawkesADM4(decay = self.decay, C = self.C, lasso_nuclear_ratio = self.lasso_nuclear_ratio, tol = self.tol, max_iter = 1000, n_threads=-1)
            
            model.fit(timestamps)
 
        elif(self.mtype == "HC"):
            model = hk.HawkesCumulantMatching(integration_support = self.integration_support, C = self.C, penalty = self.penalty, solver = self.solver, step = self.step, tol = self.tol, max_iter = 1000)
            try:
                model.fit(timestamps)
            except:
                self.error = True
        elif(self.mtype == "SG"):
            model = hk.HawkesSumGaussians(max_mean_gaussian = max_mean_gaussian, n_gaussians = n_gaussians, lasso_grouplasso_ratio = lasso_grouplasso_ratio, max_iter = 1000, n_threads=-1)
            model.fit(timestamps)
        elif(self.mtype == "Granger"):
            model = GrangerBusca(alpha_prior = 1.0/len(timestamps), num_iter = 300, beta_strategy = self.beta)
            model.fit(timestamps)
        elif(self.mtype == "ExpKern"):
            model = hk.HawkesExpKern(decays=self.decays, gofit='least-squares', penalty=self.penalty, C=self.C, solver=self.solver_exp, step=self.step, tol=self.tol, max_iter=300)
            try:
                model.fit(timestamps)
            except:
                self.error = True
        elif(self.mtype == "SumExpKern"):
            model = hk.HawkesSumExpKern(decays=[self.decays]*len(timestamps),  penalty=self.penalty, C=self.C, solver=self.solver_exp, step=self.step, tol=self.tol, max_iter=300)
            try:
                model.fit(timestamps)
            except:
                self.error = True

        self.model = model

    def score(self, data):
        timestamps = data[0]
        if(self.error == True):
            return -1e40
        if(self.mtype == "EM"):
            score = self.model.score()
            return score
        elif(self.mtype == "ADM4"):
            return self.model.score()
        elif(self.mtype == "HC"):
            return -self.model.objective(self.model.adjacency)
        elif(self.mtype == "Granger"):
            return granger_loglik(timestamps, self.model)
        elif(self.mtype == "ExpKern"):
            #print(self.decays, 'least-squares', self.penalty, self.C, self.solver_exp, self.step, self.tol, self.model.score())
            return self.model.score()
        elif(self.mtype == "SumExpKern"):
            #print(self.decays, self.penalty, self.C, self.solver_exp, self.step, self.tol, self.model.score())
            return self.model.score()



def select_hyper_EM(data, n_iter = 50):
    g = GrangerExt(mtype = "EM")
    param_dist = {
            "kernel_support": ss.uniform(1, 200), 
            "kernel_size": ss.randint(1, 400),
            "tol": ss.uniform(1e-8,1e-04)
    }
    datafortrain = (data, data) #training is training and test
    cv = ShuffleSplit(test_size = 1, n_splits = 1) 
    random_search = RandomizedSearchCV(g, param_distributions = param_dist, 
                                       n_iter = n_iter, cv = cv, verbose = 0, n_jobs = 1)
    random_search.fit(datafortrain)
    return (random_search.best_params_['kernel_support'], random_search.best_params_['kernel_size'], random_search.best_params_['tol'])

def select_hyper_ADM4(data, n_iter = 50):
    g = GrangerExt(mtype = "ADM4")
    param_dist = {
            "decay": ss.uniform(0, 9), 
            "C": ss.uniform(0, 1e4), 
            "lasso_nuclear_ratio": ss.uniform(0, 1),
            "tol": ss.uniform(1e-8,1e-04),
    }
    datafortrain = (data, data) #training is training and test
    cv = ShuffleSplit(test_size = 1, n_splits = 1) 
    random_search = RandomizedSearchCV(g, param_distributions = param_dist, 
                                       n_iter = n_iter, cv = cv, verbose = 0, n_jobs = 1)
    random_search.fit(datafortrain)
    return (random_search.best_params_['decay'], random_search.best_params_['C'],
            random_search.best_params_['lasso_nuclear_ratio'], random_search.best_params_['tol'])

def select_hyper_HC(data, n_iter = 50):
    g = GrangerExt(mtype = "HC")
    param_dist = {
        "integration_support": ss.uniform(1, 200), 
        "C": ss.uniform(0, 1e4), 
        "penalty" : ['l1', 'l2', 'none'], 
        "solver" : [ 'adam', 'adagrad', 'rmsprop', 'adadelta'], 
        "step": ss.uniform(1e-4, 1),
        "tol" : ss.uniform(1e-8,1e-4)
    }
    datafortrain = (data, data) #training is training and test
    cv = ShuffleSplit(test_size = 1, n_splits = 1) 
    random_search = RandomizedSearchCV(g, param_distributions = param_dist, 
                                       n_iter = n_iter, verbose = 0, cv = cv, n_jobs = 1)
    random_search.fit(datafortrain)
    return (random_search.best_params_['integration_support'], random_search.best_params_['C'], random_search.best_params_['penalty'], 
            random_search.best_params_['solver'], random_search.best_params_['step'], random_search.best_params_['tol'])

def select_hyper_ExpKern(data, n_iter = 50):
    g = GrangerExt(mtype = "ExpKern")
    param_dist = {
        "decays": ss.uniform(0, 9),
        "C": ss.uniform(0, 1e4), 
        "penalty" : ['l1', 'l2','nuclear','none'], 
        "solver_exp" : [ 'gd', 'agd'], 
        "step": ss.uniform(1e-5, 1),
        "tol" : ss.uniform(1e-8,1e-4)
    }
    datafortrain = (data, data) #training is training and test
    cv = ShuffleSplit(test_size = 1, n_splits = 1) 
    random_search = RandomizedSearchCV(g, param_distributions = param_dist, 
                                       n_iter = n_iter, verbose = 0, cv = cv, n_jobs = 1)
    random_search.fit(datafortrain)
    return (random_search.best_params_['decays'], random_search.best_params_['C'], random_search.best_params_['penalty'], 
            random_search.best_params_['solver_exp'], random_search.best_params_['step'], random_search.best_params_['tol'])


def select_hyper_SumExpKern(data, n_iter = 50):
    g = GrangerExt(mtype = "SumExpKern")
    param_dist = {
        "decays": ss.uniform(0, 50),
        "C": ss.uniform(0, 1e4), 
        "penalty" : ['l1', 'l2','none'], 
        "solver_exp" : [ 'gd', 'agd', 'bfgs'], 
        "step": ss.uniform(1e-4, 1e2),
        "tol" : ss.uniform(1e-12,1e-05)
    }
    datafortrain = (data, data) #training is training and test
    cv = ShuffleSplit(test_size = 1, n_splits = 1) 
    random_search = RandomizedSearchCV(g, param_distributions = param_dist, 
                                       n_iter = n_iter, verbose = 0, cv = cv, n_jobs = 1)
    random_search.fit(datafortrain)
    return (random_search.best_params_['decays'], random_search.best_params_['C'], random_search.best_params_['penalty'], 
            random_search.best_params_['solver_exp'], random_search.best_params_['step'], random_search.best_params_['tol'])


def select_hyper_Granger(data, n_iter = 150):
    g = GrangerExt(mtype = "Granger")
    param_dist = {
            "beta": ss.uniform(0.9, 5),
    }
    datafortrain = (data, data) #training is training and test
    cv = ShuffleSplit(test_size = 1, n_splits = 1, random_state = 0) 
    random_search = RandomizedSearchCV(g, param_distributions = param_dist, 
                                       n_iter = n_iter, verbose = 0, cv = cv, n_jobs = -1)
    random_search.fit(datafortrain)
    return (random_search.best_params_['beta'])