import numpy as np
from causaldag import unknown_target_igsp
import causaldag as cd
import random
# from causaldag.utils.ci_tests import gauss_ci_suffstat, gauss_ci_test, MemoizedCI_Tester
# from causaldag.utils.invariance_tests import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester
# import finite
import strategies.multi_perturbation_ed.finite as finite
from scipy.stats import entropy
from scipy.special import logsumexp
from sklearn import preprocessing
import time
import strategies.multi_perturbation_ed.main as main
import strategies.multi_perturbation_ed.mec_size as mec_size
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import sys
import math
import strategies.multi_perturbation_ed.params as params

def gen_dag_weights(dag):
    """
    input numpy matrix representation of dag, and output a weight matrix
    all weights generated uniform [-1, -0.25] or [0.25, 1]
    """
    #take a matrix of random -1s, 1s, multiply by matrix of random weights in [0,25, 1]
    #then muliplyby the dag matrix as a mask
    A = ((np.random.binomial(1, 0.5, dag.shape) - 0.5) * 2) * dag * np.random.uniform(0.25, 1, dag.shape)

    return A

def get_bs_dags(num_bs, obs_samples, nsamples_obs, nnodes, cheat_cpdag=None, bic=True):
    """
    takes in a number of bootstrap dags and observational data, outputs a list of bootstrapped dags
    cheat_dag is for debugging and doing experimets where we allow access to the MEC: on the first
    round forces the cheat cpdag into the sample
    """
    #subsample data in DAG bootstrap, and learn the DAG + MLE estimates of parameters
    bs_dags = [] # a list of the dags we get from the bootstrap
    bs_index = {} #a mapping from dag string to index in the list
    count_dags = 0 #number unique dags
    total_dags = 0 #number of dags
    samples_per_bs = nsamples_obs
    nodes = set(range(nnodes))

    while total_dags < num_bs:

        if total_dags == 0 and isinstance(cheat_cpdag,np.ndarray):
            est_cpdag = cheat_cpdag
        else:
            bs_i = np.random.choice(nsamples_obs, samples_per_bs, replace=True)
            bs_data = obs_samples[bs_i]
            #from this sample learn the DAG and an MLE of the parameters
            obs_suffstat = gauss_ci_suffstat(bs_data)
            invariance_suffstat = gauss_invariance_suffstat(obs_samples, [])
            alpha = 1e-3
            alpha_inv = 1e-3
            ci_tester = MemoizedCI_Tester(gauss_ci_test, obs_suffstat, alpha=alpha)
            invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)
            setting_list = []
            est_dag, est_targets_list = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester, nruns=5, depth=None)

            est_dag = est_dag.to_amat()[0]
            est_cpdag = main.cpdag_from_dag_observational(est_dag)

        if mec_size.mec_size(est_cpdag) <= num_bs:
            #now compute the mec and add all mec members
            mec_dags = mec_size.enumerate_dags(est_cpdag)
        else:
            #get just enough dags if mec too big
            mec_dags = mec_size.uniform_sample_dag_plural(est_cpdag, num_bs-len(bs_dags), exact=False)
        for est_dag in mec_dags:
            if est_dag.tobytes() in bs_index:
                #increase weight by one if we double count
                bs_dags[bs_index[est_dag.tobytes()]]['w'] += (1/num_bs)
                bs_dags[bs_index[est_dag.tobytes()]]['count'] += 1
                #count is the number of times the dag appears in the multiset
            else:
                A, b = finite.get_weights_MLE(est_dag, obs_samples)
                bs_dags.append({'dag': est_dag, 'A': A, 'b': b, 'w': (1/num_bs), 'count':1})
                bs_index[est_dag.tobytes()] = count_dags
                count_dags +=1
            total_dags += 1
        if total_dags > num_bs:
            break

    #for now we've just made all the weights 1/num_dags

    #now correct weights by computing the posterior of each DAG
    T = len(bs_dags)
    logPy = finite.llhood([obs_samples], [[]], bs_dags, (np.zeros(nnodes), 0)) #getting the likelihood of the observations
    #print(logPy)
    weighted_logPy = np.zeros(T)
    for i in range(T):
        #use the BIC
        weighted_logPy[i] = logPy[i] + np.log(bs_dags[i]['w']) #- np.sum(bs_dags[i]['dag']) * np.log(nsamples_obs) / 2
        if bic:
            weighted_logPy[i] = weighted_logPy[i] - np.sum(bs_dags[i]['dag']) * np.log(nsamples_obs) / 2
    denom = logsumexp(weighted_logPy)
    #now set w for each DAG to be the posterior
    for i in range(T):
        bs_dags[i]['w'] = np.exp(weighted_logPy[i] - denom)
    #remove all the tiny weights and renormalize
    w_sum = 0
    bs_dags_pruned = []
    for i in range(T):
        if bs_dags[i]['w'] >= 0.001:
            bs_dags_pruned.append(bs_dags[i])
            w_sum += bs_dags[i]['w']
    T = len(bs_dags_pruned)
    for i in range(T):
        bs_dags_pruned[i]['w'] = bs_dags_pruned[i]['w'] / w_sum
    return bs_dags_pruned

def MI_obj_gen(M, bs_dags, iv_means, K=50, iv_var = 0.01):
    """
    generates an objective function. Takes in the num synthetic datasets and the bootstrap dags,
    return a function that takes in an intervention set and returns the objective score
    K is number of repeats of each intervention
    iv_means is the mean intervention value for every node
    """
    T = len(bs_dags)
    _, _, ws = dag_cpdag_w_list(bs_dags)
    sum_ws = np.sum(ws)
    def MI_obj(epsilon, verbose=False, iter=False):
        #epsilon is a list of lists
        #to speed-up run-times, we just assume you get K samples of each intervention
        #t0= time.time()
        obj = 0
        if len(epsilon) == 0:
            return -np.inf
        for i in range(T):
            for _ in range(M):
                #sample y_mt from the intervention and compute p(y) given each possible DAG
                #first build a causaldag representation of the dag and adjacency, then sample from it
                #using the intervention
                #t1=time.time()
                cdag = cd.GaussDAG.from_amat(bs_dags[i]['A'], variances = bs_dags[i]['b'])
                #print(time.time()-t1)
                nsamples_iv = K

                #t2 = time.time()
                ivs = [{target: cd.GaussIntervention(iv_means[target], iv_var) for target in targets} for targets in epsilon]
                y_mt = [cdag.sample_interventional(iv, nsamples_iv) for iv in ivs]
                logPy = finite.llhood(y_mt, epsilon, bs_dags, (iv_means, iv_var))
                #print(time.time()-t2)

                weighted_logPy = np.zeros(T)
                for j in range(T):
                    weighted_logPy[j] = np.log(bs_dags[j]['w']/sum_ws) + logPy[j]

                #don't need compute P1 for entropies since it's constant
                #P2 is a categorical dist over DAGS
                P2 = np.zeros(T) #this will be the log dist, we'll convert after
                denom = logsumexp(weighted_logPy)
                for j in range(T):
                    P2[j] = weighted_logPy[j] - denom
                P2 = np.exp(P2) #currently just have the log dist
                if verbose:
                    print(P2)
                H2 = entropy(P2) #H2 is just the entropy induced by P2
                obj = obj - H2 * ws[i] / (M * sum_ws)
        #print(time.time()-t0)
        return obj + entropy(ws)#add prior entropy so > 0

    return MI_obj

def dist_mec(bs_dags):
    """
    given a list of bootstrapped dags, returns the list of viable MECs and a distribution of weights
    """
    cpdag_list = []
    weight_list = []
    index_dict = {} #input an MEC and get out the index
    for dag_dict in bs_dags:
        dag = dag_dict['dag']
        cpdag = main.cpdag_from_dag_observational(dag)
        cpdag_str = np.array2string(cpdag)
        if cpdag_str in index_dict:
            weight_list[index_dict[cpdag_str]] += dag_dict['w']
        else:
            index_dict[cpdag_str] = len(weight_list)
            weight_list.append(dag_dict['w'])
            cpdag_list.append(cpdag.copy())
    #print(cpdag_list)
    #print(weight_list)
    return cpdag_list, weight_list

def dag_cpdag_w_list(bs_dags):
    """
    given bs_dags, returns 3 lists, a list of dags, corresponding weight, and cpdag
    """
    cpdag_list = []
    weight_list = []
    dag_list = []
    for dag_dict in bs_dags:
        dag = dag_dict['dag']
        cpdag = main.cpdag_from_dag_observational(dag)
        weight_list.append(dag_dict['w'].copy())
        cpdag_list.append(cpdag.copy())
        dag_list.append(dag.copy())

    return dag_list, cpdag_list, weight_list

def mode_mec(bs_dags):
    """
    given a list of bootstrapped dags, returns the most likely MEC
    """
    cpdag_list, weight_list = dist_mec(bs_dags)
    return cpdag_list[np.argmax(weight_list)]


def chordal_random_finite(bs_dags, n_b, k):
    """
    Generates a chordal random intervention when there is finite data.
    Takes in the bootstrap dags list, n_b (batch size), k (num perturbations)
    """
    #sample one of our DAGs, compute the MEC (cpdag), then input into the chordal random generator
    """
    dag = bs_dags[np.random.randint(len(bs_dags), size=1)[0]]['dag']
    cpdag = main.cpdag_from_dag_observational(dag)
    """
    #cpdag = mode_mec(bs_dags)
    n = bs_dags[0]['dag'].shape[0]
    return [np.random.randint(n, size=k).tolist() for _ in range(n_b)]

def modal_cpdag(cpdags, ws):
    """
    takes in a list of cpdags and weights and returns the modal cpdag
    """
    #first construct a unique num for each cpdag
    d = {ni: indi for indi, ni in enumerate(set(cpdags))}
    numbers = [d[ni] for ni in cpdags]
    best_i = 0
    best_w = 0
    for i in range(max(numbers)+1):
        new_sum = np.sum[w[np.where(numbers==i)]]
        if new_sum > best_w:
            best_w = new_sum
            best_i = i

    return cpdags[np.where(numbers==best_i)[0]]

def union_cpdag(cpdags):
    """
    takes all cpdags in a posterior, makes them chordal, and takes the union
    """
    #only use unique cpdags
    G = np.zeros(cpdags[0].shape)
    #inefficient since don't take the set but doing this is clunky since a numpy array is not hashable
    #for very karge sets of bootstrap dags should condense the list to unique cpdags
    for cpdag in cpdags:
        G = np.maximum(G + np.minimum(cpdag, 0), -1) #the min makes the cpdag chordal
        #the max ensure edges aren't double counted
    return G

def ss_finite(bs_dags, obj, n_b, k, smart_ss, lazy=True, all_k=False):
    """
    seperating system approach on finite dataset
    """
    #sample a dag, compute the cpdag, then do infinite sample approach on finite sample obj
    """
    dag = bs_dags[np.random.randint(len(bs_dags), size=1)[0]]['dag']
    cpdag = main.cpdag_from_dag_observational(dag)
    """
    #given the current method of finding a sep system, the cpdag inputted doesn't matter
    n = (bs_dags[0]['dag']).shape[0]
    _, cpdags, _ = dag_cpdag_w_list(bs_dags)
    if lazy:
        return main.lazy_ss_intervention(n, n_b, k, obj, union_cpdag(cpdags), smart_ss, all_k=all_k)
    else:
        return main.ss_intervention(n, n_b, k, obj, union_cpdag(cpdags), smart_ss)

def ss_infinite(bs_dags, obj, n_b, k, smart_ss, num_samples=10, lazy=True, all_k=False):
    """
    seperating system method using infinite sample objective
    """
    #cpdags, ws = dist_mec(bs_dags)
    #new_obj = main.edge_obj_sample(cpdags, ws, num_samples, obj=obj)
    #changed to use just the same dags as the finite sample setting with the correct weights
    dags, cpdags, ws = dag_cpdag_w_list(bs_dags)
    #print(cpdags)
    new_obj, _ , _= main.weighted_dags_edge_obj_sample_ss(cpdags, ws, dags, obj = obj)
    n = (bs_dags[0]['dag']).shape[0]
    if lazy:
        return main.lazy_ss_intervention(n, n_b, k, new_obj, union_cpdag(cpdags), smart_ss, all_k=all_k)
    else:
        return main.ss_intervention(n, n_b, k, new_obj, union_cpdag(cpdags), smart_ss)

def gred_infinite(bs_dags, n_b, k, obj):
    """
    seperating system method using infinite sample objective
    """
    dags, cpdags, ws = dag_cpdag_w_list(bs_dags)
    new_obj, new_stochastic_grad, new_hess_fun = main.weighted_dags_edge_obj_sample(cpdags, ws, dags, obj = None)
    n = (bs_dags[0]['dag']).shape[0]
    #just input the first cpdag- its not used
    return main.scdpp(cpdags[0], n_b, k, new_obj, new_stochastic_grad, new_hess_fun, T=5, max_score=1, M0 = 5, M=5)

def drg_infinite(bs_dags, n_b, k):
    """
    discrete random greedy on infinite sample obj
    """
    dags, cpdags, ws = dag_cpdag_w_list(bs_dags)
    new_obj, _,_ = main.weighted_dags_edge_obj_sample(cpdags, ws, dags, obj = None)
    n = (bs_dags[0]['dag']).shape[0]
    return main.lazy_drg(n, n_b, k, new_obj)


def ss_infinite_cont(bs_dags, n_b, k, obj):
    """
    continuous seperating system approach
    """
    dags, cpdags, ws = dag_cpdag_w_list(bs_dags)
    new_obj, new_stochastic_grad, new_hess_fun = main.weighted_dags_edge_obj_sample_ss(cpdags, ws, dags, obj = None)
    n = (bs_dags[0]['dag']).shape[0]
    return main.scdpp_ss(union_cpdag(cpdags), n_b, k, new_obj, new_stochastic_grad, new_hess_fun, T=5*n_b, max_score=1, M0 = 5, M=5)


def process_score(inter, bs_dags, loss, OVs, meth, f, k, b, k_range, inters_dict):
    obj_val = loss(inter)
    #stores the first box index that obtains more objective value
    #corresponds to the number of random single perturbation interventions
    #needed to outperform the method
    #step_score = np.argmax(np.array(score_boxes)>=obj_val).item()
    #if using ss_a take the better score with lower k if it exists
    if meth in ['ss_a', 'ss_inf_a']:
        for k_p in k_range:
            if k_p < k:
                f_p = "b=" + str(b) + '_k=' + str(k_p) +'_'+meth
                #step_score = max(step_score, steps[f_p][-1])
                obj_val = max(obj_val, OVs[f_p][-1])
                inter = inters_dict[f_p][-1]
    f = "b=" + str(b) + '_k=' + str(k) +'_'+meth
    OVs[f].append(obj_val)
    inters_dict[f].append(inter)

    print(obj_val)


def run_finite_experiment(nnodes, generator, k_range, meths, labs, title = '', name='', repeats=10, do_plot=True, nsamples_obs=1000, K=20, M=20, num_bs=10):
    """
    runs experiments on finite obs samples. everything is linear gaussian
    nnodes: num nodes in graph
    nsamples_obs: the number of samples of observational data
    K: number of samples per intervention you give
    M:samples per simulation in the objective
    num_bs: the number of bootstrapped dags
    """

    #to start all the same as in infinite sample case
    OVs = {}
    times_dict = {}
    true_orient = {}
    steps={}
    dag_dict = []
    inters_dict = {}

    fig = plt.figure()
    b_range = [1, 2, 3, 4, 5, 6, 7, 8]
    plt.xticks(b_range)

    lines = ['-', '--', ':']
    invalid_list = []
    true_dags = []
    initial_dags_list = []

    for _ in range(repeats):
        valid_dag = False
        while valid_dag == False:
            #print("in while loop")
            #print(generator)
            random_root = np.random.choice(nnodes)
            if generator == 'chain':
                dag = main.generate_chain_dag_fixed_root(nnodes, random_root)
            elif generator == 'tree':
                dag = main.uniform_random_tree(nnodes)
            elif generator == "bipartite":
                dag = main.ER_bipartite(int(nnodes/2), nnodes- int(nnodes/2), 0.5)
            elif generator.startswith('ER'):
                #for erdos renyi write ER plus the param value
                rho = float(generator.split('_')[1])
                dag=main.generate_ER(nnodes, rho)
            elif generator == "fully_connected":
                dag = main.generate_fully_connected(nnodes)
            elif generator.startswith("barabasi_albert"):
                #for barabasi-albert append "_m" to the string
                m = int(generator.split('_')[2])
                dag = main.generate_barabasi_albert(nnodes, m)
            else:
                raise Exception
            #print(dag)
            cpdag = main.cpdag_from_dag_observational(dag)
            #print(cpdag)
            max_score = main.cpdag_obj_val(dag) - main.cpdag_obj_val(cpdag) # for normalizing scores
            #accept dags with mec size less than 100 and more than 5
            #print(max_score)
            mec_size_dag = mec_size.mec_size(cpdag)
            if  mec_size_dag <=200 and mec_size_dag >= nnodes:
                valid_dag = True
            else:
                invalid_list.append(1)

        print(cpdag)


        #now that we have a dag, generate weights and convert to cd format
        d = cd.DAG.from_amat(dag)
        g = cd.rand.rand_weights(d) #all variances are 1 if not stated in the func call
        #we use causaldag package to generate but could use our methods and convert the type
        # Generate observational data
        obs_samples = g.sample(nsamples_obs)
        true_dag = {'dag':dag, 'A': g._weight_mat, 'b': np.ones(1)}
        true_dags.append(true_dag.copy())

        bs_dags = get_bs_dags(num_bs, obs_samples, nsamples_obs, nnodes, cheat_cpdag=cpdag)
        initial_dags_list.append(bs_dags.copy())
        #now bs_dags has a list of all the learnt dags
        print(bs_dags)
        #compute loss function given this set of DAGs and parameters
        loss = MI_obj_gen(M, bs_dags, np.zeros(nnodes)+5, K=K) #interventions of size 5

        score_boxes = [loss([])]
        inters = []

        for k in k_range:
            for b in b_range:
                for i in range(len(meths)):
                    meth = meths[i]
                    print(meth)
                    f = "b=" + str(b) + '_k=' + str(k) +'_'+meth
                    #create the entry if the list is not in the dict yet
                    if f not in OVs:
                        OVs[f] = []
                        times_dict[f] = []
                        inters_dict[f] = []
                    #do these last methods all at once
                    if meth in ['ss_inf_a', 'ss_inf_b', 'ss_a', 'ss_b', 'cont'] and b != b_range[-1]:
                        continue
                    start_time = time.perf_counter()
                    if meth == 'rand':
                        inter = chordal_random_finite(bs_dags, b, k)
                    elif meth == 'ss_inf_a':
                        inter=ss_infinite(bs_dags, "MI", b, k, False, all_k = False)
                        print(inter)
                    elif meth == 'ss_inf_b':
                        inter=ss_infinite(bs_dags, "MI", b, k, True)
                    elif meth == 'ss_a':
                        ss_loss = MI_obj_gen(int(M/10), bs_dags,np.zeros(nnodes)+5, K=K)
                        inter=ss_finite(bs_dags, ss_loss, b, k, False, all_k=False)
                    elif meth == 'ss_b':
                        ss_loss = MI_obj_gen(int(M/10), bs_dags,np.zeros(nnodes)+5, K=K)
                        inter=ss_finite(bs_dags, ss_loss, b, k, True)
                    elif meth == 'cont':
                        inter=gred_infinite(bs_dags, b, k, None)
                    elif meth == 'ss_inf_b_cont':
                        inter=ss_infinite_cont(bs_dags, b, k, None)
                    else:
                        raise Exception

                    times_dict[f].append(time.perf_counter()-start_time)
                    print(times_dict[f][-1])

                    #for the greedy methods just compute everything in one go
                    if meth in ['ss_inf_a', 'ss_inf_b', 'ss_a', 'ss_b', 'cont']:
                        for bp in b_range:
                            inter_p = inter[0:bp]
                            process_score(inter_p, bs_dags, loss, OVs, meth, f, k, bp, k_range, inters_dict)

                        continue

                    process_score(inter, bs_dags, loss, OVs, meth, f, k, b, k_range, inters_dict)

    if do_plot:
        for k in k_range:
            for i in range(len(meths)):
                meth = meths[i]
                bmean = []
                ebar= []
                times = []
                for b in b_range:
                    f = "b=" + str(b) + '_k=' + str(k) +'_'+meth
                    bmean.append(np.mean(OVs[f]))
                    ebar.append(np.std(OVs[f])/math.sqrt(repeats))
                    times.append(np.mean(times_dict[f]))

                plt.figure(1)
                plt.errorbar(b_range, bmean, ebar, color= params.colors[i], linestyle=lines[k-1], label = "rand, k=" +str(k))

                plt.figure(2)
                plt.plot(b_range, times, color=params.colors[i], linestyle=lines[k-1], label = "rand, k=" +str(k))

    if do_plot:
        num_p = len(meths)
        custom_lines = []
        for i in range(num_p):
            custom_lines.append(Line2D([0], [0], color=params.colors[i], lw=4))

        custom_lines = custom_lines + [Line2D([0], [0], color='black', lw=2), Line2D([0], [0], color='black', lw=2, linestyle="--"), Line2D([0], [0], color='black', lw=2, linestyle=":")]

        plt.figure(1)

        plt.xticks(b_range)

        plt.xlabel('Batch Size')
        plt.ylabel('Normalized Objective')
        plt.legend(custom_lines, labs+['k=' + str(k) for k in k_range], loc='lower right')

        plt.title(title)
        plt.savefig(name + '.pdf', bbox_inches='tight')

        plt.figure(2)
        #integer xticks
        plt.xticks(b_range)
        plt.legend(custom_lines, labs+ ['k=' + str(k) for k in k_range], loc='lower right')
        plt.title("Average Time per run")
        plt.xlabel('Batch Size')
        plt.ylabel('Seconds')
        plt.savefig(name + '_time.pdf', bbox_inches='tight')

        #clear and close both figures
        plt.clf()
        plt.close()

        plt.figure(1)
        plt.clf()
    plt.close()

    #save plot data in json
    with open(name +'_OVs.json', 'w') as fp:
        json.dump(OVs, fp)
    with open(name + '_times.json', 'w') as fp:
        json.dump(times_dict, fp)

    with open(name + '_inters_dict.json', 'w') as fp:
        json.dump(inters_dict, fp)

    for i in range(len(true_dags)):
        true_dags[i]['A'] = true_dags[i]['A'].tolist()
        true_dags[i]['b'] = true_dags[i]['b'].tolist()
        true_dags[i]['dag'] = true_dags[i]['dag'].tolist()

    with open(name + '_true_dags.json', 'w') as fp:
        json.dump(true_dags, fp)

    for i in range(len(initial_dags_list)):
        for j in range(len(initial_dags_list[i])):
            initial_dags_list[i][j]['A'] = initial_dags_list[i][j]['A'].tolist()
            initial_dags_list[i][j]['b'] = initial_dags_list[i][j]['b'].tolist()
            initial_dags_list[i][j]['dag'] = initial_dags_list[i][j]['dag'].tolist()

    with open(name + '_initial_dag_list.json', 'w') as fp:
        json.dump(initial_dags_list, fp)

    #list of invalid dag tallies
    with open(name + '_invalids.json', 'w') as fp:
        json.dump(invalid_list, fp)

    return

if __name__ == '__main__':
    # Generate a true DAG and parameters
    np.random.seed(42)
    #first command is run id
    if len(sys.argv) > 1:
        run = int(sys.argv[1])
        np.random.seed(run)
    else:
        run = 0
        np.random.seed(42)

    meths = ['rand', 'ss_inf_b', 'cont',  'ss_inf_a', 'ss_a', 'ss_b']
    labs = ['rand',  'ss_inf_b', 'cont', 'ss_inf_a', 'ss_a', 'ss_b']

    k_range = [1,2,3,4,5]
    generators = ['ER_0.1']
    repeat_list = [1]
    for nnodes in [20,40]:
        for i in range(len(repeat_list)):
            generator = generators[i]
            title ="Mean Objective Value on " + generator + " Graph n=" + str(nnodes),
            name ='figures/' + generator + "_n=" + str(nnodes) + '_' + str(run)
            run_finite_experiment(nnodes, generator, k_range, meths, labs, title = title, name=name, repeats=repeat_list[i], do_plot=False, nsamples_obs=nnodes*50, K=3, M=10, num_bs=100)








