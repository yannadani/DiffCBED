"""
The module for all experiments related to the finite sample case
"""

import numpy as np
import networkx as nx
import strategies.multi_perturbation_ed.main as main
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal
from scipy.stats import norm
import time

#functions needed
#takes a graph and returns a weight matrix
#returns a noise matrix
#generates data from the data generating process
#given the data, subsamples it via dag bootstrap
#dag learning algo by wang 17
#method to fit mle of parameters given the dag

def gen_weights(dag):
    """
    takes in a dag, returns a weight matrix
    #do uniform random on [-1, 1]. Anything in [-0.25, 0.25] clips to 0.
    """
    n = dag.shape[0]

    A_sign = (np.random.randint(2,size=(n,n)) -0.5)*2

    #generate 0.25 to 1, then multiply by random sign, then by the DAG entries
    A = np.random.uniform(0.25,1,size=(n,n)) * A_sign * dag
    return A

def gen_noise_param(n):
    """
    takes in the number of nodes, returns the noise of each node
    """
    #for now just have stdev=1 for every node
    return np.ones(n)*0.1

def get_ordering(dag):
    """
    takes in a dag and returns the topological ordering as a list
    """
    g = nx.DiGraph(dag)
    order = list(nx.topological_sort(g))
    return order

def get_weights_MLE(dag, obs_data):
    """
    given a dag and some observational data, return the weights and noises given by the MLE
    assumes linear gaussian
    """
    #get the ordering and then do linear regressions
    pi = get_ordering(dag)
    n = dag.shape[0]
    A = np.zeros((n,n))
    b = np.zeros(n)
    for i in range(n):
        deps = np.flatnonzero(dag[:, i]) #all the arrows entering the node
        #if no dependants just compute noise term
        if len(deps) == 0:
            b[i] = np.var(obs_data[:, i])
        else:
            X = obs_data[:, deps]
            y = obs_data[:, i]
            reg = LinearRegression(fit_intercept = False).fit(X, y)
            for j in range(len(deps)):
                A[deps[j]][i] = reg.coef_[j]
            b[i] = np.var(y - reg.predict(X))
    return A, b

def get_weight_MLE_int(dag, obs_data, inter_data, inters):
    """
    given dag and observational plus interventional data, get the MLE of weights
    """
    #get the ordering and then do linear regressions
    pi = get_ordering(dag)
    n = dag.shape[0]
    A = np.zeros((n,n))
    b = np.zeros(n)
    for i in range(n):
        deps = np.flatnonzero(dag[:, i]) #all the arrows entering the node
        #if no dependants just compute noise term
        if len(deps) == 0:
            b[i] = np.var(obs_data[:, i])
        else:
            X = obs_data[:, deps]
            y = obs_data[:, i]
            for I_index in range(len(inter_data)):
                I = inters[I_index]
                #ignore when we were perturbed
                if i in I:
                    continue
                X = np.concatenate([X, inter_data[I_index][:, deps]], axis=0)
                y = np.concatenate([y, inter_data[I_index][:, i]])
            reg = LinearRegression(fit_intercept = False).fit(X, y)
            for j in range(len(deps)):
                A[deps[j]][i] = reg.coef_[j]
            b[i] = np.var(y - reg.predict(X))
    return A, b

def normalize_combined(obs_data, inter_data, inters):
    """
    normalize so that data is zero mean
    """
    n = len(obs_data[0])
    means = np.zeros(n)
    for i in range(n):
        y = obs_data[:, i]
        for I_index in range(len(inter_data)):
            I = inters[I_index]
            #ignore when we were perturbed
            if i in I:
                continue
            y = np.concatenate([y, inter_data[I_index][:, i]])
        means[i] = np.mean(y)
    for inter_i in range(len(inter_data)):
        inter_data[inter_i] = inter_data[inter_i] - means
    return obs_data - means, inter_data, -means

def llhood(data, interventions, dag_dicts, int_stats):
    """
    data consists of a list of intervention data, interventions the list of interventions,
    and we also input a list of dag dictionaries. int_stats is just a tuple: the mean (for
    each node) and variance of the gaussian perfect intervention
    outputs a numpy array of the the log likelihoods of the data given the interventions and each dag_dict
    """
    #compute the conditional mean and variance of each node for each row
    #then calculate the likelihood using posterior of a multivariate gaussian

    out = []

    for dag_dict in dag_dicts:
        means = [np.matmul(dt, dag_dict['A']) for dt in data]
        #now correct for the interventions
        for i in range(len(means)):
            intervention = interventions[i]
            for perturbation in intervention:
                means[i][:, perturbation] = int_stats[0][perturbation] #set all intervened columns
        var_list = [dag_dict['b'].copy() for dt in data]
        for i in range(len(var_list)):
            intervention = interventions[i]
            for perturbation in intervention:
                var_list[i][perturbation] = int_stats[1] #set all intervened columns
        #now just compute the prob of the multivariate gaussian
        logpdf = 0
        flat_data = None
        flat_mean = None
        flat_var = None
        flat_std = None
        #print("time1")
        t0 = time.time()
        for i in range(len(means)):
            #treat each dist seperately since its faster

            if flat_data is None:
                flat_data = data[i].flatten()
                flat_mean = means[i].flatten()
                flat_var = np.tile(var_list[i], data[i].shape[0])
            else:
                flat_data = np.concatenate((flat_data, data[i].flatten()), axis=0)
                flat_mean = np.concatenate((flat_mean, means[i].flatten()), axis=0)
                flat_var = np.concatenate((flat_var, np.tile(var_list[i], data[i].shape[0])), axis=0)
            #so using scikit is slow- i do my own computation instead
            """
            for j in range(flat_data.shape[0]):
                logpdf += norm.logpdf(flat_data[j], flat_mean[j], np.sqrt(flat_var[j]))
            """
        flat_std = np.sqrt(flat_var)
        const = len(flat_mean) * -np.log(np.sqrt(2 * np.pi))
        logpdf += -0.5 * np.sum(np.square((flat_data-flat_mean)/flat_std)) - np.sum(np.log(flat_std)) + const
        #this is another way of computing it that is the slowest
        #multivariate_normal.logpdf(data[i].flatten(), mean=means[i].flatten(), cov=np.tile(var_list[i], data[i].shape[0]), allow_singular=True)
        out.append(logpdf)

    return np.asarray(out)

def gen_data(m, A, b, T):
    """
    takes in weight matrix A and noise matrix b, generates data as a linear SEM, T is the ordering
    int m is dataset size. in final output each row is an observation, each column a node
    """
    #first generate the noise then go through the ordering applying the right column
    #of A each time

    n= b.size

    epsilon = np.random.normal(0, b, size=(m, n))
    out = epsilon
    #go through T and set each node to be function of its parents
    for i in T:
        out[:, i] = out[:, i] + np.matmul(A[:,i], out.T) #add back on the noise
    return out



