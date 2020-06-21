import numpy as np
import tick.hawkes as hk
from random_search import *
import pp_metrics
import aux_funcs
from abc import ABC, abstractmethod
import sklearn.preprocessing
import sys
import subprocess
from sklearn.model_selection import ParameterSampler
from gb import GrangerBusca


class PointProcessModel(ABC):
    @abstractmethod
    def fit(self):
        raise NotImplementedException
    @abstractmethod
    def optimize_params(self):
        raise NotImplementedException
    @abstractmethod
    def get_kernel_norms(self):
        raise NotImplementedException
    @abstractmethod
    def get_baselines(self):
        raise NotImplementedException
    @abstractmethod
    def score(self, timestamps):
        raise NotImplementedException
    @abstractmethod
    def set_params(self):
        raise NotImplementedException
    @abstractmethod
    def read_params(self):
        raise NotImplementedException
    @abstractmethod
    def get_params(self):
        raise NotImplementedException

class HawkesProcess(PointProcessModel):
    def fit(self,timestamps):
        self.model.fit(timestamps)
    def get_kernel_norms(self):
        return self.model.get_kernel_norms()
    def get_baselines(self):
        return self.model.baseline
    def score(self, timestamps):
        return self.model.score()

class HkEM(HawkesProcess):
    def __init__(self, kernel_support=10, kernel_size=10, tol=1e-05, max_iter=300):
        self.name = "HkEM"
        self.hawkes = True
        self.kernel_support = kernel_support
        self.kernel_size = kernel_size
        self.tol = tol
        self.max_iter = max_iter
        self.model = hk.HawkesEM(kernel_support=self.kernel_support, kernel_size=self.kernel_size, tol=self.tol, max_iter=self.max_iter, n_threads=-1)
        self.param_dist = {
            "kernel_support": ss.uniform(1, 200), 
            "kernel_size": ss.randint(1, 400),
            "tol": ss.uniform(1e-8,1e-04)
        }
    def get_params(self):
        return (self.kernel_support, self.kernel_size, self.tol)
    def set_params(self,kernel_support, kernel_size, tol):
        self.kernel_support, self.kernel_size, self.tol = kernel_support, kernel_size, tol
        self.model = hk.HawkesEM(kernel_support=self.kernel_support, kernel_size=self.kernel_size, tol=self.tol, max_iter=self.max_iter, n_threads=-1)
    def optimize_params(self, timestamps, n_iter=60):
        self.kernel_support, self.kernel_size, self.tol = select_hyper_EM(timestamps, n_iter = n_iter)
        self.model = hk.HawkesEM(kernel_support=self.kernel_support, kernel_size=self.kernel_size, tol=self.tol, max_iter=self.max_iter, n_threads=-1)
    def read_params(self,filename):
        self.kernel_support, self.kernel_size, self.tol = eval(open(filepath,"r").readline())
        self.model = hk.HawkesEM(kernel_support=self.kernel_support, kernel_size=self.kernel_size, tol=self.tol, max_iter=self.max_iter, n_threads=-1)

class ADM4(HawkesProcess):
    def __init__(self, decay=1, C=1000, lasso_nuclear_ratio=0.5, tol=1e-05, max_iter=300):
        self.name = "ADM4"
        self.hawkes = True
        self.decay = decay
        self.C = C
        self.lasso_nuclear_ratio = lasso_nuclear_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.model = hk.HawkesADM4(decay=self.decay, C=self.C, lasso_nuclear_ratio=self.lasso_nuclear_ratio, tol=self.tol, max_iter=self.max_iter, n_threads=-1)
        self.param_dist = {
            "decay": ss.uniform(0, 10), 
            "C": ss.uniform(0, 1e3), 
            "lasso_nuclear_ratio": ss.uniform(0, 1),
            "tol": ss.uniform(1e-8,1e-04),
        }
    def get_params(self):
        return (self.decay, self.C, self.lasso_nuclear_ratio, self.tol)
    def set_params(self,decay, C, lasso_nuclear_ratio, tol):
        self.decay, self.C, self.lasso_nuclear_ratio, self.tol = decay, C, lasso_nuclear_ratio, tol
        self.model = hk.HawkesADM4(decay=self.decay, C=self.C, lasso_nuclear_ratio=self.lasso_nuclear_ratio, tol=self.tol, max_iter=self.max_iter, n_threads=-1)
    def optimize_params(self, timestamps, n_iter=75):
        self.decay, self.C, self.lasso_nuclear_ratio, self.tol = select_hyper_ADM4(timestamps, n_iter = n_iter)
        self.model = hk.HawkesADM4(decay=self.decay, C=self.C, lasso_nuclear_ratio=self.lasso_nuclear_ratio, tol=self.tol, max_iter=self.max_iter, n_threads=-1)
    def read_params(self,filename):
        self.decay, self.C, self.lasso_nuclear_ratio, self.tol = eval(open(filepath,"r").readline())
        self.model = hk.HawkesADM4(decay=self.decay, C=self.C, lasso_nuclear_ratio=self.lasso_nuclear_ratio, tol=self.tol, max_iter=self.max_iter, n_threads=-1)

class SG(HawkesProcess):
    def __init__(self, max_mean_gaussian=100, n_gaussians=10, step_size=1e-7, C=1e3, lasso_grouplasso_ratio=0.5, tol=1e-05, max_iter=300):
        self.name = "SG"
        self.hawkes = True
        self.max_mean_gaussian = max_mean_gaussian
        self.n_gaussians = n_gaussians
        self.step_size = step_size
        self.C = C
        self.lasso_grouplasso_ratio = lasso_grouplasso_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.model = hk.HawkesSumGaussians(max_mean_gaussian=self.max_mean_gaussian, n_gaussians=self.n_gaussians, step_size=self.step_size, C=self.C, lasso_grouplasso_ratio=self.lasso_grouplasso_ratio, max_iter=self.max_iter, n_threads=-1)
        self.param_dist = {
            "max_mean_gaussian": ss.uniform(0, 200), 
            "n_gaussians": ss.randint(1, 20),
            "step_size": ss.uniform(1e-9, 1e-3),
            "C": ss.uniform(0, 1e3),
            "lasso_grouplasso_ratio": ss.uniform(0, 1),
            "tol": ss.uniform(1e-8,1e-04)
        }
    def get_params(self):
        return (self.max_mean_gaussian, self.n_gaussians, self.step_size, self.C, self.lasso_grouplasso_ratio, self.tol)
    def set_params(self,max_mean_gaussian, n_gaussians, step_size, C, lasso_grouplasso_ratio, tol):
        self.max_mean_gaussian, self.n_gaussians, self.step_size, self.C, self.lasso_grouplasso_ratio, self.tol = max_mean_gaussian, n_gaussians, step_size, C, lasso_grouplasso_ratio, tol
        self.model = hk.HawkesSumGaussians(max_mean_gaussian=self.max_mean_gaussian, n_gaussians=self.n_gaussians, step_size=self.step_size, C=self.C, lasso_grouplasso_ratio=self.lasso_grouplasso_ratio, max_iter=self.max_iter, n_threads=-1)    
    def optimize_params(self, timestamps, n_iter=75):
        raise NotImplementedException
    def read_params(self,filename):
        self.max_mean_gaussian, self.n_gaussians, self.step_size, self.C, self.lasso_grouplasso_ratio, self.tol = eval(open(filepath,"r").readline())
        self.model = hk.HawkesSumGaussians(max_mean_gaussian=self.max_mean_gaussian, n_gaussians=self.n_gaussians, step_size=self.step_size, C=self.C, lasso_grouplasso_ratio=self.lasso_grouplasso_ratio, max_iter=self.max_iter, n_threads=-1)
    def score(self, timestamps):
        return None

class BasisKernels(HawkesProcess):
    def __init__(self, kernel_support=10, n_basis=None, kernel_size=10,C=1e-1,tol=1e-05, max_iter=300):
        self.name = "BasisKernels"
        self.hawkes = True
        self.kernel_support = kernel_support
        self.n_basis = n_basis
        self.kernel_size = kernel_size
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.model = hk.HawkesBasisKernels(kernel_support=self.kernel_support, n_basis=self.n_basis, kernel_size=self.kernel_size,C=self.C,tol=self.tol,max_iter=self.max_iter, n_threads=-1)
        self.param_dist = {
            "kernel_support": ss.uniform(1, 200), 
            "n_basis": ss.randint(1, 100),
            "kernel_size": ss.randint(1, 400),
            "C": ss.uniform(1e-3, 10),
            "tol": ss.uniform(1e-8,1e-04)
        }
    def get_params(self):
        return (self.kernel_support,self.n_basis,self.kernel_size,self.C,self.tol)
    def set_params(self,kernel_support,n_basis,kernel_size,C,tol):
        self.kernel_support,self.n_basis,self.kernel_size,self.C,self.tol = kernel_support,n_basis,kernel_size,C,tol
        self.model = hk.HawkesBasisKernels(kernel_support=self.kernel_support, n_basis=self.n_basis, kernel_size=self.kernel_size,C=self.C,tol=self.tol,max_iter=self.max_iter, n_threads=-1)    
    def fit(self,timestamps):
        self.model = hk.HawkesBasisKernels(kernel_support=self.kernel_support, n_basis=len(timestamps), kernel_size=self.kernel_size,C=self.C,tol=self.tol,max_iter=self.max_iter, n_threads=-1)
        self.model.fit(timestamps)
    def get_kernel_norms(self):
        return np.einsum('ijk->ij', self.model.amplitudes)
    def optimize_params(self, timestamps, n_iter=75):
        raise NotImplementedException     
    def read_params(self,filename):
        self.kernel_support,self.n_basis,self.kernel_size,self.C,self.tol = eval(open(filepath,"r").readline())
        self.model = hk.HawkesBasisKernels(kernel_support=self.kernel_support, n_basis=self.n_basis, kernel_size=self.kernel_size,C=self.C,tol=self.tol,max_iter=self.max_iter, n_threads=-1)
    def score(self, timestamps):
        return None

class CondLaw(HawkesProcess):
    def __init__(self):
        self.name = "CondLaw"
        self.hawkes = True
        self.model = hk.HawkesConditionalLaw(n_threads=-1)
    def optimize_params(self, timestamps, n_iter=75):
        raise NotImplementedException
    def read_params(self,filename):
        self.model = hk.HawkesConditionalLaw(n_threads=-1)
    def get_params(self):
        raise NotImplementedException
    def set_params(self):
        raise NotImplementedException
    def score(self, timestamps):
        return None

class HC(HawkesProcess):
    def __init__(self, integration_support=50, C=1000, penalty='none', solver='adagrad', step=1, tol=1e-05, max_iter=300):
        self.name = "Cumulant"
        self.hawkes = True
        self.integration_support = integration_support
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.step = step
        self.tol  = tol 
        self.max_iter = max_iter
        self.model = hk.HawkesCumulantMatching(integration_support=self.integration_support, C=self.C, penalty=self.penalty, solver=self.solver, step=self.step, tol=self.tol,max_iter=self.max_iter)
        self.param_dist = {
            "integration_support": ss.uniform(1, 200), 
            "C": ss.uniform(0, 1e3), 
            "penalty" : ['l1', 'l2', 'none'], 
            "solver" : [ 'adam', 'adagrad', 'rmsprop', 'adadelta'], 
            "step": ss.uniform(1e-4, 1),
            "tol" : ss.uniform(1e-8,1e-4)
        }
    def get_params(self):
        return (self.integration_support, self.C, self.penalty, self.solver, self.step, self.tol)
    def set_params(self,integration_support, C, penalty, solver, step, tol):
        self.integration_support, self.C, self.penalty, self.solver, self.step, self.tol = integration_support, C, penalty, solver, step, self.tol
        self.model = hk.HawkesCumulantMatching(integration_support=self.integration_support, C=self.C, penalty=self.penalty, solver=self.solver, step=self.step, tol=self.tol,max_iter=self.max_iter)
    def optimize_params(self, timestamps, n_iter=200):
        self.integration_support, self.C, self.penalty, self.solver, self.step, self.tol = select_hyper_HC(timestamps, n_iter = n_iter)
        self.model = hk.HawkesCumulantMatching(integration_support=self.integration_support, C=self.C, penalty=self.penalty, solver=self.solver, step=self.step, tol=self.tol,max_iter=self.max_iter)
    def score(self, timestamps):
        return -self.model.objective(self.model.adjacency)
    def read_params(self,filename):
        self.integration_support, self.C, self.penalty, self.solver, self.step, self.tol = eval(open(filepath,"r").readline())
        self.model = hk.HawkesCumulantMatching(integration_support=self.integration_support, C=self.C, penalty=self.penalty, solver=self.solver, step=self.step, tol=self.tol,max_iter=self.max_iter)


class ExpKern(HawkesProcess):
    def __init__(self, decays=1, gofit='least-squares', penalty='none', C=1000, solver='gd', step=1, tol=1e-05, max_iter=300):
        self.name = "ExpKern"
        self.hawkes = True
        self.decays = decays
        self.gofit = gofit
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.step = step
        self.tol = tol
        self.max_iter = max_iter 
        self.model = hk.HawkesExpKern(decays=self.decays,  gofit=self.gofit, penalty=self.penalty,  C=self.C,  solver=self.solver,  step=self.step,  tol=self.tol,  max_iter=self.max_iter)
        self.param_dist = {
            "decays": ss.uniform(0, 9),
            "C": ss.uniform(0, 1e3), 
            "penalty" : ['l1', 'l2','nuclear','none'], 
            "solver_exp" : [ 'gd', 'agd'], 
            "step": ss.uniform(1e-5, 1),
            "tol" : ss.uniform(1e-8,1e-4)
        }
    def get_params(self):
        return (self.decays, self.C, self.penalty, self.solver, self.step, self.tol)
    def set_params(self,decays, C, penalty, solver, step, tol):
        self.decays, self.C, self.penalty, self.solver, self.step, self.tol = decays, C, penalty, solver, step, self.tol
        self.model = hk.HawkesExpKern(decays=self.decays,  gofit=self.gofit, penalty=self.penalty,  C=self.C,  solver=self.solver,  step=self.step,  tol=self.tol,  max_iter=self.max_iter)
    def optimize_params(self, timestamps, n_iter=50):
        self.decays, self.C, self.penalty, self.solver_exp, self.step, self.tol = select_hyper_ExpKern(timestamps, n_iter = n_iter)
        self.model = hk.HawkesExpKern(decays=self.decays,  gofit=self.gofit, penalty=self.penalty,  C=self.C,  solver=self.solver,  step=self.step,  tol=self.tol,  max_iter=self.max_iter)
    def read_params(self,filename):
        self.decays, self.C, self.penalty, self.solver_exp, self.step, self.tol = eval(open(filepath,"r").readline())
        self.model = hk.HawkesExpKern(decays=self.decays,  gofit=self.gofit, penalty=self.penalty,  C=self.C,  solver=self.solver,  step=self.step,  tol=self.tol,  max_iter=self.max_iter)

class Wold(PointProcessModel):
    def __init__(self, beta=1, num_iter=300):
        self.name = "Granger"
        self.hawkes = False
        self.beta = beta
        self.num_iter = num_iter
        self.model = GrangerBusca(alpha_prior=1.0/100, num_iter=self.num_iter, beta_strategy = self.beta)
        self.param_dist = {
            "beta": ss.uniform(0.9, 5),
        }
    def get_params(self):
        return (self.beta, self.num_iter)
    def set_params(self,beta):
        self.beta = beta
        self.model = GrangerBusca(alpha_prior=1.0/100, num_iter = 300, beta_strategy = self.beta)
    def optimize_params(self, timestamps, n_iter=150):
        self.beta = select_hyper_Granger(timestamps, n_iter)
        self.model = GrangerBusca(alpha_prior=1.0/len(timestamps), num_iter=self.num_iter, beta_strategy = self.beta)
    def fit(self,timestamps):
        self.model = GrangerBusca(alpha_prior=1.0/len(timestamps), num_iter=self.num_iter, beta_strategy = self.beta)
        self.model.fit(timestamps)
    def get_kernel_norms(self):
        return sklearn.preprocessing.normalize(self.model.Alpha_.toarray(),"l1")
    def get_baselines(self):
        return self.model.mu_
    def score(self, timestamps):
        return pp_metrics.granger_loglik(timestamps, self.model)
    def read_params(self,filename):
        self.beta = eval(open(filepath,"r").readline())
        self.model = GrangerBusca(alpha_prior=1.0/100, num_iter = 300, beta_strategy = self.beta)


class netinf(PointProcessModel):
    def __init__(self, alpha=1, dist=0, num_iter=300):
        self.name = "NetInf"
        self.hawkes = False
        self.alpha = alpha
        self.dist = dist
        self.num_iter = num_iter
        self.param_dist = {
            "alpha": ss.uniform(0.1, 1),
            "dist": ss.randint(0, 3) #exp, powerlaw, rayleigh
        }
    def get_params(self):
        return (self.alpha, self.dist)
    def set_params(self,alpha,dist):
        self.alpha = alpha
        self.dist = dist
    def optimize_params(self, timestamps, n_iter=150):
        param_score = []
        for param_set in ParameterSampler(self.param_dist, n_iter=n_iter):
            self.set_params(param_set["alpha"], param_set["dist"])
            self.fit(timestamps)
            param_score.append((self.get_params(),self.score(timestamps)))
        param_score = sorted(param_score, key=lambda x:x[1])
        self.set_params(param_score[0][0][0],param_score[0][0][1])
    def fit(self,timestamps):
        filepath = "../cascades/netinf/"
        bashCommand = filepath+"./netinf -i:"+filepath+"netinf_memetracker.txt -e:"+str(self.num_iter)+\
                    " -a:"+str(self.alpha)+" -m:"+str(self.dist)+" -s:2 -o:"+filepath+"network"
        
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if output.find(b"rror") != -1:
            print("ERROR netinf")
            print(output)
            raise
        n_nodes = len(timestamps)
        self.save_kernel_norms(filepath,n_nodes)
        self.save_obj(filepath)
    def save_kernel_norms(self,filepath,n_nodes):
        self.norms = np.zeros((n_nodes,n_nodes))
        with open(filepath+"network.txt") as f:
            for line in f:
                nodes = line.strip("\n").split(',')
                if len(nodes) > 1:
                    node1 = int(nodes[0])
                    node2 = int(nodes[1])
                    self.norms[node1][node2] = 1.
    def save_obj(self,filepath):
        with open(filepath+"network-objective.tab") as f:
            for line in f:
                lt = line.strip("\n").split("\t")
                if len(lt) == 2 and lt[0]!="# Iters":
                    niter = int(lt[0])
                    obj_value = float(lt[1])
        assert niter <= self.num_iter
        self.obj_value = obj_value
    def get_kernel_norms(self):
        return self.norms
    def get_baselines(self):
        return None
    def score(self, timestamps):
        return self.obj_value
    def read_params(self,filename):
        self.alpha, self.dist = eval(open(filepath,"r").readline())