import json
import params
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
import numpy as np
import csv
import dream
import sklearn.metrics 
from scipy.special import logsumexp
import causaldag as cd
import finite
from sklearn import preprocessing
import copy
import matplotlib.ticker as ticker

def load_exp_data(name, title, runs, k_range, to_plot, finite=False, var_names=['rand', 'gred', 'ss'], legend_names=['Random', 'GrED', 'SS'], x_axis = "b", dream=False, dream_i = 0, b_range=[1,2,3,4,5], mid_legend = False):
    """
    Loads in data from different parallel runs and makes the plots of objective achieved vs k or b
    Used for both infinite sample and finite samples
    input:
    str name: filename
    str title: what to call the plot
    int runs: number of repeats to cycle through
    bool finite: whether these a finite sample results
    var_names: the titles of the different files
    legend_names: the corresponding legend text for each method
    output:
    None, produces some plots
    """
    num_p = len(var_names) #number of plotted lines
    OVs = [] #OV = objective value
    times = []

    indices = (np.arange(runs)+1).tolist()
    #correction of indices for the dream experiments where we do 5 runs of each
    if dream:
        indices = ((np.arange(runs))*5 + dream_i + 1).tolist()

    for run in indices:
        name1 = name + '_' + str(run)
        with open(name1 + '_OVs.json', 'r') as fp:
            OVs.append(json.load(fp))
        with open(name1 + '_times' + '.json', 'r') as fp:
            times.append(json.load(fp))

    #now plot the figures
    fig = plt.figure()
    plt.xticks(b_range)
    
    lines = ['-', '--', ':', '-.', 'None'][0:len(k_range)]

    bmean = {}
    ebar = {}
    tmean = {}
    for vn in var_names:
        bmean[vn] = np.zeros((len(k_range), len(b_range)))
        ebar[vn] = np.zeros((len(k_range), len(b_range)))
        tmean[vn] = np.zeros((len(k_range), len(b_range)))


        for k_ind in range(len(k_range)):
            k=k_range[k_ind]
            for b_ind in range(len(b_range)):
                b = b_range[b_ind]
                f = "b=" + str(b) + '_k=' + str(k)
                OV = []
                time = []
                for run in range(len(indices)):
                    OV.append(OVs[run][f+'_'+vn])
                    time.append(times[run][f+'_'+vn])
                bmean[vn][k_ind, b_ind] = np.mean(OV)
                ebar[vn][k_ind, b_ind] = np.std(OV)/math.sqrt(len(OV))
                #time mean throws errors right now since we don't get times for all b
                tmean[vn][k_ind, b_ind] = np.mean(time)


    #now we iterate through the tuples of vn, k pairs we want on the plot
    plt.figure(1)
    line_num = 0
    markers = ["o", "<", "*", "D", "X", ">", 7]
    if x_axis == "b":
        for i in range(len(to_plot)):
            setting = to_plot[i]
            var = setting[0]
            k_ind = setting[1]
            k = k_range[k_ind]
            leg = legend_names[i]
            plt.errorbar(b_range, bmean[var][k_ind,:], ebar[var][k_ind,:], color= params.colors[i], linestyle=lines[2], marker=markers[line_num], label = leg) 
            line_num += 1
    else:
        for i in range(len(to_plot)):
            setting = to_plot[i]
            var = setting[0]
            b_ind = setting[1]
            b = b_range[b_ind]
            leg = legend_names[i]
            plt.errorbar(k_range, bmean[var][:,b_ind], ebar[var][:,b_ind], color= params.colors[i], linestyle=lines[2],marker=markers[line_num], label = leg) 
            line_num += 1
    #print(gred_OV)
    plt.figure(2)
    if x_axis=='b':
        
        #names are longer so reformat x labels
        plt.xticks(rotation=60)
        time_plot = []
        for var in to_plot:
            time_plot.append(tmean[var[0]][var[1], -1])
        print(legend_names)
        print(time_plot)
        
    
    plt.figure(1)
    custom_lines = []
    num_p = len(legend_names)
    for i in range(num_p):
        custom_lines.append(Line2D([0], [0], color=params.colors[i], lw=2, marker=markers[i]))
    #don't add different k to the legend now
    #custom_lines = custom_lines + [Line2D([0], [0], color='black', lw=2), Line2D([0], [0], color='black', lw=2, linestyle="--"), Line2D([0], [0], color='black', lw=2, linestyle=":")]

    if mid_legend:
        plt.legend(custom_lines, legend_names, loc='right', prop={"size":16})
    else:
        plt.legend(custom_lines, legend_names, loc='lower right', prop={"size":16})
    if x_axis == "b":
        plt.xlabel('Batch Size (m)', fontsize= 20)
    else:
        plt.xlabel('Intervention Size (q)', fontsize= 20)
    if dream:
        plt.ylabel('Proportion of Edges Oriented', fontsize= 20)
    elif finite:
        plt.ylabel(r'MI Objective $F_{MI}$', fontsize= 20)
    else:
        plt.ylabel(r'Edge Orientation Objective $F_{EO}$', fontsize= 20)
    #plt.title(title)
    if x_axis=="k":
        name_file = name + "_kaxis"
    else:
        name_file = name

    plt.tick_params(labelsize=14)

    plt.savefig(name_file + '.pdf', bbox_inches='tight')
    
    plt.figure(2)
    if x_axis=='b':
        #plt.legend(custom_lines, legend_names + ['k=' + str(k) for k in k_range], loc='lower right')
        #plt.title("Average Time per run")
        plt.ylabel('Average Runtime (Seconds)')
        plt.tick_params(labelsize=16)
        plt.bar(legend_names, time_plot, color="green")
        plt.savefig(name + '_time.pdf', bbox_inches='tight')
    
    plt.figure(1)
    plt.clf()
    plt.close()
    plt.figure(2)
    plt.clf()
    plt.close()

    #print out how many invalid graphs you got
    total_invalid = 0
    for run in indices:
        name1 = name + '_' + str(run)
        with open(name1 + '_invalids' + '.json', 'r') as fp:
            total_invalid += np.sum(json.load(fp))
    print("total invalid for " + name + ":")
    print(total_invalid) #this was the number of times we had to regenerate the DAG
    return

def posterior(epsilon, bs_dags, true_dag_dict, iv_means, iv_var, K):
    """
    computes the posterior given some interventions by sampling data
    from the true dag and using the list of bootstrapped dags as the support
    of the posterior
    """
    #read interventional data in
    T= len(bs_dags)
    # Generate observational data
    g = cd.GaussDAG.from_amat(np.asarray(true_dag_dict['A']))
    nsamples_iv = K

    ivs = [{target: cd.GaussIntervention(iv_means[target], iv_var) for target in targets} for targets in epsilon]
    y = [g.sample_interventional(iv, nsamples_iv) for iv in ivs] 

    #convert epsilon to numpy
    logPy = finite.llhood(y, epsilon, bs_dags, (iv_means, iv_var))
     
    weighted_logPy = np.zeros(T)
    for j in range(T):
        weighted_logPy[j] = np.log(bs_dags[j]['w']) + logPy[j]
    
    P2 = np.zeros(T) #this will be the log dist, we'll convert after
    denom = logsumexp(weighted_logPy)
    for j in range(T):
        P2[j] = weighted_logPy[j] - denom
    P2 = np.exp(P2) #currently just have the log dist
    for j in range(T):
        bs_dags[j]['w'] = P2[j]
    return bs_dags

def f1_score(bs_dags, true_dag):
    """
    compute the F1 score for predicting prescence of
    a directed edge in a DAG
    true_dag is the dag itself not the dict
    """
    n = true_dag.shape[0]

    #first compute the probability of each edge
    prob_g = np.zeros((n, n))
    for bs_dag in bs_dags:
        prob_g = prob_g + bs_dag['w'] * np.asarray(bs_dag['dag'])

    #for each threshold we compute precion, accuracy, F
    thresholds = np.linspace(0,1,100)
    F = 0
    for thresh in thresholds:
        est_dag = (prob_g > thresh).astype(int)
        F_temp = sklearn.metrics.f1_score(np.ndarray.flatten(true_dag), np.ndarray.flatten(est_dag))
        #report the highest F score
        if F_temp > F:
            F = F_temp
    return F

def mean_shd(bs_dags, true_dag):
    """
    computes the average shd between the posterior over DAGs and the true DAG
    """
    out = 0
    c = 0 #normalization constant
    for bs_dag in bs_dags:
        out += bs_dag['w'] * np.sum(bs_dag['dag'] != true_dag)
        c += bs_dag['w']

    return out / c

def finite_process(name, title, n, runs, k_range, var_names=['rand', 'gred', 'ss'], legend_names=['Random', 'GrED', 'SS']):
    """
    on finite experiments, compute SHD and F1 scores
    """
    K=3 #number of samples per experiment
    b_range = [1, 2, 3,4,5,6,7,8]
    f1s = {}
    f1s_ebar = {}
    shds = {}
    shds_ebar = {}
    for vn in var_names:
        f1s[vn] = np.zeros((len(k_range), len(b_range)))
        shds[vn] = np.zeros((len(k_range), len(b_range)))
        f1s_ebar[vn] = np.zeros((len(k_range), len(b_range)))
        shds_ebar[vn] = np.zeros((len(k_range), len(b_range)))
    #for each k, b, method, get the posterior distributions, compute f1 and shd
    for k_ind in range(len(k_range)):
        k = k_range[k_ind]
        for b_ind in range(len(b_range)):
            b = b_range[b_ind]
            for vn in var_names:
                true_dag_list = []
                bs_dags_list = []
                inters_list = []
                f1_list = []
                shd_list = []
            
                f = "b=" + str(b) + '_k=' + str(k) +'_'+vn
                for run in range(1, runs+1):
                    name_vn = name + '_' + str(run)
                    #read in the true dags, the interventions, dag dists
                    with open(name_vn + '_true_dags.json', 'r') as fp:
                        true_dag_list = true_dag_list + json.load(fp)
                    with open(name_vn + '_initial_dag_list.json', 'r') as fp:
                        bs_dags_list = bs_dags_list + json.load(fp)
                        #this is method specific
                    with open(name_vn + '_inters_dict.json', 'r') as fp:
                        inters_list = inters_list + json.load(fp)[f]
                temp_bs_dags_list = copy.deepcopy(bs_dags_list)
                num_repeats = 10
                for i in range(len(true_dag_list)):
                    f1_val = 0
                    shd_val = 0
                    for _ in range(num_repeats):
                        post_dag = posterior(inters_list[i], copy.deepcopy(temp_bs_dags_list[i]), true_dag_list[i], np.zeros(n)+5, 0.01, K=K)
                        print(f1_score(post_dag, np.asarray(true_dag_list[i]['dag'])))
                        f1_val += f1_score(post_dag, np.asarray(true_dag_list[i]['dag'])) / num_repeats
                        shd_val += mean_shd(post_dag, np.asarray(true_dag_list[i]['dag'])) / num_repeats
                    f1_list.append(f1_val)
                    shd_list.append(shd_val)

                f1s[vn][k_ind, b_ind] = np.mean(f1_list)
                shds[vn][k_ind, b_ind] = np.mean(shd_list)
                f1s_ebar[vn][k_ind, b_ind] = np.std(f1_list) / math.sqrt(len(f1_list))
                shds_ebar[vn][k_ind, b_ind] = np.std(shd_list) / math.sqrt(len(shd_list))
    #for saving, convert to list
    f1s_list = {}
    shds_list = {}
    f1s_ebar_list = {}
    shds_ebar_list = {}
    for vn in var_names:
        f1s_list[vn] = f1s[vn].tolist()
        shds_list[vn] = shds[vn].tolist()
        f1s_ebar_list[vn] = f1s_ebar[vn].tolist()
        shds_ebar_list[vn] = shds_ebar[vn].tolist()
    with open(name + '_f1s.json', 'w') as fp:
        json.dump(f1s_list, fp)
    with open(name + '_shds.json', 'w') as fp:
        json.dump(shds_list, fp)
    with open(name + '_f1s_ebar.json', 'w') as fp:
        json.dump(f1s_ebar_list, fp)
    with open(name + '_shds_ebar.json', 'w') as fp:
        json.dump(shds_ebar_list, fp)

    return f1s, shds, f1s_ebar, shds_ebar

def read_finite(name, n, runs, k_range, var_names):
    """
    ran before plotting F1 and SHD scores if saved from 
    finite_process
    read in results from existing jsons, convert to numpy
    """
    with open(name + '_f1s.json', 'r') as fp:
        f1s = json.load(fp)
    with open(name + '_shds.json', 'r') as fp:
        shds = json.load(fp)
    with open(name + '_f1s_ebar.json', 'r') as fp:
        f1s_ebar = json.load(fp)
    with open(name + '_shds_ebar.json', 'r') as fp:
        shds_ebar = json.load(fp)

    for vn in var_names:
        f1s[vn] = np.asarray(f1s[vn])
        shds[vn] = np.asarray(shds[vn])
        f1s_ebar[vn] = np.asarray(f1s_ebar[vn])
        shds_ebar[vn] = np.asarray(shds_ebar[vn])

    return f1s, shds, f1s_ebar, shds_ebar


def finite_plot(name, k_range, f1s, shds, f1s_ebar, shds_ebar, to_plot, legend_names, x_axis="b", b_range=[1,2,3,4,5,6,7,8]):
    """
    creates plots for the finite experiment's SHD and F1 scores
    """

    is_k = int(x_axis == "k")

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    #ensures integer xticks
    if x_axis=='b':
        plt.xticks(b_range)
    else:
        plt.xticks(k_range)

    fig4, ax4 = plt.subplots()
    if x_axis=='b':
        plt.xticks(b_range)
    else:
        plt.xticks(k_range)

    lines = ['-', '--', ':', '-.', 'None'][0:len(k_range)]

    plt.figure(1+2*is_k)
    line_num = 0
    markers = ["o", "<", "*", "D", "X", ">", 7, 6]
    if x_axis == "b":
        for i in range(len(to_plot)):
            setting = to_plot[i]
            var = setting[0]
            k_ind = setting[1]
            k = k_range[k_ind]
            leg = legend_names[i]
            plt.errorbar(b_range, f1s[var][k_ind,:], f1s_ebar[var][k_ind,:], color= params.colors[i], linestyle=lines[2], marker=markers[line_num], label = leg) 
            line_num += 1
    else:
        for i in range(len(to_plot)):
            setting = to_plot[i]
            var = setting[0]
            b_ind = setting[1]
            b = b_range[b_ind]
            leg = legend_names[i]
            plt.errorbar(k_range, f1s[var][:,b_ind], f1s_ebar[var][:,b_ind], color= params.colors[i], linestyle=lines[2],marker=markers[line_num], label = leg) 
            line_num += 1
    #plot shd
    plt.figure(2+2*is_k)
    line_num = 0
    if x_axis == "b":
        for i in range(len(to_plot)):
            setting = to_plot[i]
            var = setting[0]
            k_ind = setting[1]
            k = k_range[k_ind]
            leg = legend_names[i]
            plt.errorbar(b_range, shds[var][k_ind,:], shds_ebar[var][k_ind,:], color= params.colors[i], linestyle=lines[2], marker=markers[line_num], label = leg) 
            line_num += 1
    else:
        for i in range(len(to_plot)):
            setting = to_plot[i]
            var = setting[0]
            b_ind = setting[1]
            b = b_range[b_ind]
            leg = legend_names[i]
            plt.errorbar(k_range, shds[var][:,b_ind], shds_ebar[var][:,b_ind], color= params.colors[i], linestyle=lines[2],marker=markers[line_num], label = leg) 
            line_num += 1

    for plot_i in [1+is_k*2,2+is_k*2]:
        plt.figure(plot_i)
        custom_lines = []
        num_p = len(legend_names)
        for i in range(num_p):
            custom_lines.append(Line2D([0], [0], color=params.colors[i], lw=2, marker=markers[i]))
        if plot_i%2 == 1:
            plt.legend(custom_lines, legend_names, loc='lower right', prop={"size":16})
        else:
            plt.legend(custom_lines, legend_names, loc='upper right', prop={"size":16})
        if x_axis == "b":
            plt.xlabel('Batch Size (m)', fontsize= 20)
        else:
            plt.xlabel('Intervention Size (q)', fontsize= 20)

        if plot_i%2 == 1:
            plt.ylabel('F1 Score', fontsize= 20)
        else:
            plt.ylabel('Structural Hamming Distance', fontsize= 20)

        plt.tick_params(labelsize=14)
        #plt.title(title)
        if x_axis=="k":
            name_file = name + "_kaxis"
        else:
            name_file = name

        if plot_i%2 == 1:
            name_file = name_file + "_f1"
        else:
            name_file = name_file + "_shd"

        plt.savefig(name_file + '.pdf', bbox_inches='tight')


 
    plt.clf()
    plt.close()

    return


#infinite sample experiments plot num edges identified on average
to_plot = [('ss_a', 2), ('ss_b', 2), ('cont', 2), ('rand', 2), ('ss_a', 0)]
labs = ["SSGa (q=3)", "SSGb (q=3)", "DGC (q=3)", 'Rand (q=3)', 'Greedy (q=1)']

for n in [10, 20, 40]:
        runs = 50

        if n > 20:
            meths = ['rand', 'ss_a', 'ss_b', 'ss_a_cont','ss_b_cont', 'cont', 'drg'] 
            k_range = [1, 2, 3, 4, 5]
        else:
            meths = ['rand', 'ss_a', 'ss_b', 'ss_a_cont','ss_b_cont', 'cont', 'drg'] 
            k_range = [1, 2, 3]
        
        name ='figures_feb14/tree_n=' + str(n)
        title = "Mean Objective Value on Tree Graph n=" + str(n)
        load_exp_data(name, title, runs, k_range, to_plot, var_names=meths, legend_names=labs)

        name ='figures_feb14/ER_0.25_n=' + str(n)
        title = "Mean Objective Value on ER 0.25 Graph n=" + str(n)
        load_exp_data(name, title, runs, k_range, to_plot,var_names=meths, legend_names=labs)

        name ='figures_feb14/ER_0.1_n=' + str(n)
        title = "Mean Objective Value on ER_0.1 Graph n=" + str(n)
        load_exp_data(name, title, runs, k_range, to_plot,var_names=meths, legend_names=labs)

to_plot = [('ss_a', 2), ('ss_b', 2), ('cont', 2), ('rand', 2), ('ss_a', 0)]
labs = ["SSGa (q=3)", "SSGb (q=3)", "DGC (q=3)", 'Rand (q=3)', 'Greedy (q=1)']
k_range = [1,2,3]
for n in [10,20]:
    name ='figures_feb14/star_n=' + str(n)
    title = "Mean Objective Value on K-Star Graph n=" + str(n)
    load_exp_data(name, title, runs, k_range, to_plot,var_names=meths, legend_names=labs)
    
    name ='figures_feb14/ER_0.5_n=' + str(n)
    title = "Mean Objective Value on ER_0.5 Graph n=" + str(n)
    load_exp_data(name, title, runs, k_range, to_plot,var_names=meths, legend_names=labs)


to_plot = [('ss_a', 1), ('ss_b', 1), ('cont', 1), ('rand', 1)]
labs = ["SSGa (m=2)", "SSGb (m=2)", "DGC (m=2)", "Rand (m=2)"]
n=5
k_range = [1, 2, 3]
name ='figures_feb14/fully_connected_n=' + str(n)
title = "Mean Objective Value on Fully Connected Graph n=" + str(n)
load_exp_data(name, title, runs, k_range, to_plot,var_names=meths, legend_names=labs,  x_axis="k")

k_range = [1,2,3,4,5]
n=40
to_plot = [('ss_a', 2), ('ss_b', 2), ('cont', 2), ('rand', 2)]
labs = ["SSGa (m=3)", "SSGb (m=3)", "DGC (m=3)", 'Rand (m=3)']
name ='figures_feb14/tree_n=' + str(n)
title = "Mean Objective Value on Tree Graph n=" + str(n)
load_exp_data(name, title, runs, k_range, to_plot, var_names=meths, legend_names=labs, x_axis="k")


#process DREAM experiments 
meths = ['rand', 'ss_a', 'ss_b', 'cont'] 
n=50
k_range = [1, 2, 3, 4, 5]
runs=5
for i in range(1,6):
    name ='figures_dream_feb16/dream_' + str(i) + '_n=' + str(n)
    title = "Mean Objective Value on Dream Graph n=" + str(n)
    to_plot = [('ss_a', 4), ('ss_b', 4), ('cont', 4), ('rand', 4), ('ss_a', 1)]
    labs = ["SSGa (q=5)", "SSGb (q=5)", "DGC (q=5)", 'Rand (q=5)', 'Greedy (q=1)']
    load_exp_data(name, title, runs, k_range, to_plot,var_names=meths, legend_names=labs, dream=True, dream_i=i-1)
    to_plot = [('ss_a', 2), ('ss_b', 2), ('cont', 2), ('rand', 2)]
    labs = ["SSGa (m=3)", "SSGb (m=3)", "DGC (m=3)", 'Rand (m=3)']
    load_exp_data(name, title, runs, k_range, to_plot,var_names=meths, legend_names=labs,  x_axis="k", dream=True, dream_i=i-1)


#finite plots on direct objective

runs = 200
for n in [40]:
    meths = ['rand', 'ss_inf_b', 'cont',  'ss_inf_a', 'ss_a', 'ss_b']
    if n == 40:
        k_range = [1,2,3,4,5]
    else:
        k_range = [1,2,3]

    to_plot = [('ss_inf_b', 4), ('ss_b', 4), ('cont', 4), ('rand', 4), ('ss_inf_a', 4), ('ss_a', 4), ('ss_a', 0)]
    labs = [r'SSGb-$\infty$ (q=5)', "SSGb (q=5)", r'DGC-$\infty$ (q=5)', 'Rand (q=5)',  r'SSGa-$\infty$ (q=5)', "SSGa (q=5)", 'Greedy (q=1)']
    name ='figures_finite_feb24/ER_0.1_n=' + str(n)
    title = "Mean Objective Value on ER 0.1 Graph n=" + str(n)
    load_exp_data(name, title, runs, k_range, to_plot, finite=True, var_names=meths, legend_names=labs, b_range=[1,2,3,4,5,6,7,8])

    to_plot = [('ss_inf_b', 1), ('ss_b', 1), ('cont', 1), ('rand', 1), ('ss_inf_a', 1), ('ss_a', 1)]
    labs = [r'SSGb-$\infty$ (m=2)', "SSGb (m=2)", r'DGC-$\infty$ (m=2)', 'Rand (m=2)', r'SSGa-$\infty$ (m=2)', "SSGa (m=2)"]
    name ='figures_finite_feb24/ER_0.1_n=' + str(n)
    title = "Mean Objective Value on ER 0.1 Graph n=" + str(n)
    load_exp_data(name, title, runs, k_range, to_plot, finite=True, var_names=meths, legend_names=labs, x_axis="k", b_range=[1,2,3,4,5,6,7,8], mid_legend=True)


#finite plots F1 and SHD
runs = 200
k_range = [1,2,3,4,5]
for n in [40]:
    meths = ['rand', 'ss_inf_b', 'cont',  'ss_inf_a', 'ss_a', 'ss_b']#, 'ss_inf_b_cont']

    to_plot = [('ss_inf_b', 4), ('ss_b', 4), ('cont', 4), ('rand', 4), ('ss_inf_a', 4), ('ss_a', 4), ('ss_a', 0)]
    labs = [r'SSGb-$\infty$ (q=5)', "SSGb (q=5)", r'DGC-$\infty$ (q=5)', 'Rand (q=5)', r'SSGa-$\infty$ (q=5)', "SSGa (q=5)", 'Greedy (q=1)']
    name ='figures_finite_feb24/ER_0.1_n=' + str(n)
    title = "Mean F1 on ER 0.1 Graph n=" + str(n)

    #comment out read_finite if not yet computed F1 or SHD scores, comment out finite_process if already
    #have these computed. finite_process computes these and saves them, read_finite just reads them if they 
    # have already been computed
    #f1s, shds, f1s_ebar, shds_ebar = finite_process(name, title, n, runs, k_range, var_names=meths, legend_names=labs)

    f1s, shds, f1s_ebar, shds_ebar = read_finite(name, n, runs, k_range, meths)

    finite_plot(name, k_range, f1s, shds, f1s_ebar, shds_ebar, to_plot, labs, x_axis="b")

    to_plot = [('ss_inf_b', 1), ('ss_b', 1), ('cont', 1), ('rand', 1), ('ss_inf_a', 1), ('ss_a', 1)]
    labs = [r'SSGb-$\infty$ (m=2)', "SSGb (m=2)", r'DGC-$\infty$ (m=2)', 'Rand (m=2)', r'SSGa-$\infty$ (m=2)', "SSGa (m=2)"]
    finite_plot(name, k_range, f1s, shds, f1s_ebar, shds_ebar, to_plot, labs, x_axis="k")



#all finite plots again without ssga

runs = 200
for n in [40]:
    meths = ['rand', 'ss_inf_b', 'cont', 'ss_a', 'ss_b']
    if n == 40:
        k_range = [1,2,3,4,5]
    else:
        k_range = [1,2,3]

    to_plot = [('ss_inf_b', 4), ('ss_b', 4), ('cont', 4), ('rand', 4), ('ss_a', 0)]
    labs = [r'SSGb-$\infty$ (q=5)', "SSGb (q=5)", r'DGC-$\infty$ (q=5)', 'Rand (q=5)', 'Greedy (q=1)']
    name ='figures_finite_feb24_no_ssga/ER_0.1_n=' + str(n)
    title = "Mean Objective Value on ER 0.1 Graph n=" + str(n)
    load_exp_data(name, title, runs, k_range, to_plot, finite=True, var_names=meths, legend_names=labs, b_range=[1,2,3,4,5,6,7,8])

    to_plot = [('ss_inf_b', 1), ('ss_b', 1), ('cont', 1), ('rand', 1)]
    labs = [r'SSGb-$\infty$ (m=2)', "SSGb (m=2)", r'DGC-$\infty$ (m=2)', 'Rand (m=2)']
    name ='figures_finite_feb24_no_ssga/ER_0.1_n=' + str(n)
    title = "Mean Objective Value on ER 0.1 Graph n=" + str(n)
    load_exp_data(name, title, runs, k_range, to_plot, finite=True, var_names=meths, legend_names=labs, x_axis="k", b_range=[1,2,3,4,5,6,7,8], mid_legend=True)


runs = 200
k_range = [1,2,3,4,5]
for n in [40]:
    meths = ['rand', 'ss_inf_b', 'cont', 'ss_b', 'ss_a']#, 'ss_inf_b_cont']

    to_plot = [('ss_inf_b', 4), ('ss_b', 4), ('cont', 4), ('rand', 4), ('ss_a', 0)]
    labs = [r'SSGb-$\infty$ (q=5)', "SSGb (q=5)", r'DGC-$\infty$ (q=5)', 'Rand (q=5)', 'Greedy (q=1)']
    name ='figures_finite_feb24_no_ssga/ER_0.1_n=' + str(n)
    title = "Mean F1 on ER 0.1 Graph n=" + str(n)

    #comment out read_finite if not yet computed F1 or SHD scores, comment out finite_process if already
    #have these computed
    #f1s, shds, f1s_ebar, shds_ebar = finite_process(name, title, n, runs, k_range, var_names=meths, legend_names=labs)

    f1s, shds, f1s_ebar, shds_ebar = read_finite(name, n, runs, k_range, meths)

    finite_plot(name, k_range, f1s, shds, f1s_ebar, shds_ebar, to_plot, labs, x_axis="b")

    to_plot = [('ss_inf_b', 1), ('ss_b', 1), ('cont', 1), ('rand', 1)]
    labs = [r'SSGb-$\infty$ (m=2)', "SSGb (m=2)", r'DGC-$\infty$ (m=2)', 'Rand (m=2)']
    finite_plot(name, k_range, f1s, shds, f1s_ebar, shds_ebar, to_plot, labs, x_axis="k")

