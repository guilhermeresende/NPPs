import numpy as np
from gb import simulate,GrangerBusca
import tick.hawkes as hk
from sklearn.preprocessing import normalize

def gen_alpha1(d):
    Alpha=np.zeros((d,d))
    for i in range(0,d):
        for j in range(0,d):
            if(i < d//2):
                if(j >= d//2 and j <= d-1-i):
                    Alpha[i][j] = 2/d
            else:
                if(j < d//2 and j >= d-1-i):
                    Alpha[i][j] = 2/d

    quarter_pt = d//4
    Alpha[quarter_pt][quarter_pt] = 2/d
    Alpha[quarter_pt][quarter_pt+1] = 2/d
    Alpha[quarter_pt+1][quarter_pt] = 2/d
    Alpha[quarter_pt+1][quarter_pt+1] = 2/d

    Alpha = normalize(Alpha,'l1')
    return Alpha

def gen_alpha2(d):
    Alpha = np.zeros((d,d))
    for i in range(0,d):
        for j in range(0,d):
            if(i < d//5):
                if(j >= d-1-i):
                    Alpha[i][j] = 2/d
            elif(i < d//2):
                if(j >= d//2 and j <= d-1-i):
                    Alpha[i][j] = 2/d
            else:
                if(j < d//2 and j >= d-1-i):
                    Alpha[i][j] = 2/d
    Alpha = normalize(Alpha,'l1')
    return Alpha

def gen_alpha3(d, n_clusters):
    assert d%n_clusters == 0
    div_num = d/n_clusters
    Alpha = np.zeros((d,d))
    for i in range(0,d):
        for j in range(0,d):
            if i//div_num == j//div_num:
                Alpha[i][j] = 1./div_num
    Alpha = normalize(Alpha,'l1')
    return Alpha

def gen_alpha4(d):
    Alpha = np.zeros((d, d))

    for i in range(d//2):
        for j in range(d//2):
            if i >= j:
                Alpha[i][j] = 2/d

    for i in range(d//2, d):
        for j in range(d//2, d):
            if i >= j:
                Alpha[i][j] = 2/d
    Alpha = normalize(Alpha,'l1')
    return Alpha

def simulate_hawkes_exp(n_points, d, alphaShape="alpha1", n_clusters=None):
    if (alphaShape == "alpha1"):
        Alpha = gen_alpha1(d)
    elif (alphaShape == "alpha2"):
        Alpha = gen_alpha2(d)
    elif (alphaShape == "alpha3"):
        Alpha = gen_alpha3(d, n_clusters)
    elif (alphaShape == "alpha4"):
        Alpha = gen_alpha4(d)

    Beta = np.ones(shape = (d,d))
    mu = np.ones(shape = d) * 0.1

    kernels = [[hk.HawkesKernelExp(Alpha[i][j],Beta[i][j]) for j in range(len(Alpha[i]))] for i in range(len(Alpha))]
    h = hk.SimuHawkes(kernels=kernels, baseline=list(mu), max_jumps=n_points)
    h.simulate()
    timestamps = h.timestamps

    return (timestamps, Alpha, mu)

def simulate_hawkes_plaw(n_points, d, alphaShape="alpha1", n_clusters=None):
    if (alphaShape == "alpha1"):
        Alpha = gen_alpha1(d)
    elif (alphaShape == "alpha2"):
        Alpha = gen_alpha2(d)
    elif (alphaShape == "alpha3"):
        Alpha = gen_alpha3(d, n_clusters)
    elif (alphaShape == "alpha4"):
        Alpha = gen_alpha4(d)

    Beta = np.ones(shape = (d,d)) * 2
    mu = np.ones(shape = d) * 0.1

    kernels = [[hk.HawkesKernelPowerLaw  (Alpha[i][j], 1.5, Beta[i][j]) for j in range(len(Alpha[i]))] for i in range(len(Alpha))]
    h = hk.SimuHawkes(kernels=kernels, baseline=list(mu), max_jumps=n_points)
    h.simulate()
    timestamps = h.timestamps

    return (timestamps, Alpha, mu)

def simulate_granger(n_points, d, alphaShape="alpha1", n_clusters=None):
    if (alphaShape == "alpha1"):
        Alpha = gen_alpha1(d)
    elif (alphaShape == "alpha2"):
        Alpha = gen_alpha2(d)
    elif (alphaShape == "alpha3"):
        Alpha = gen_alpha3(d, n_clusters)
    elif (alphaShape == "alpha4"):
        Alpha = gen_alpha4(d)

    Alpha=Alpha.T
    Beta = np.ones(shape = (d,d)) 
    mu = np.ones(shape = d) * 0.1
    sim = simulate.GrangeBuscaSimulator(mu_rates = mu, Alpha_ba = Alpha, Beta_ba = Beta)
    ticks_granger = sim.simulate(n_events=n_points)
    ticks=([np.array(p) for p in ticks_granger])

    return (ticks, Alpha, mu)
