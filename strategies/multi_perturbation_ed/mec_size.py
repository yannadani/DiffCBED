import numpy as np
import networkx as nx
import math
from networkx.algorithms.clique import find_cliques as maximal_cliques
#from networkx.algorithms.tree.decomposition import junction_tree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import itertools
from itertools import combinations
from networkx.algorithms.tree.recognition import is_tree as nx_is_tree
import strategies.multi_perturbation_ed.main
import time


def is_fully_connected(G):
    #True if graph G is fully connected
    return np.all(G - (np.eye(G.shape[0], G.shape[0])) == -1)

def is_tree(G):
    """
    True if G a tree or forest
    """
    G_nx = nx.Graph(G*-1)
    return nx_is_tree(G_nx)

def size_he_cc(G):
    """
    He et al. 2015: to break a connected component into all its possible dags
    n is the number of nodes
    node_ids is the ids of the nodes in the connected component
    """

    #turn it to undirected (set edges as -1s)
    #for each possible root, orient edges and compute size
    G = (G*(-1)).astype(int)
    size = 0
    #if its oriented just return it
    if np.all(G >= 0):
        size += 1
        return size

    #if its fully connected return it
    if is_fully_connected(G):
        size += math.factorial(G.shape[0])
        return size

    #if its a tree return it
    if is_tree(G):
        return G.shape[0]

    for v in range(G.shape[0]):
        G_v = G.copy()
        #orient all edges as out of v
        G_v[v, :] = G_v[v,:] * -1
        G_v[:, v] = 0
        new_edge_end = np.where(G_v[v,:] == 1)
        new_edges = []
        for out_node in new_edge_end[0]:
            new_edges.append((v,out_node))
        G_v = main.meek(G_v, new_edges=new_edges, skip_r3=False, is_tree=False)
        #if we get a DAG just add it on
        if np.all(G_v >= 0):
            size += 1
        else:
            #otherwise run it through size again
            size_1 = size_he(G_v)
            size += size_1

    return size

def size_he(G):
    """
    computes the size of MEC given by G
    similar to He et al. 2015
    """

    #break G into its connected components and return the product
    #remove directed edges from G
    G = np.minimum(G, 0)

    #for this purpose represent all edges as directed now
    G = (G*(-1)).astype(int)

    graph = csr_matrix(G)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    G_size = 1
    #break G into connected components and product their sizes
    for i in range(n_components):
        index = np.nonzero(labels == i)[0] #the 0 index into first element of tuple
        size1 = size_he_cc(G[index][:, index])
        G_size = G_size * size1
    return G_size

def mec_size(G):
    """
    shell function for interfacing with other modules
    computes the MEC size of a graph G.
    """
    return size_he(G.copy())

def is_fully_directed(G):
    """
    takes in graph in directed edge format and returns true if all edges directed
    """
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i][j] == 1 and G[j][i] == 1:
                return False
    return True

def fast_sample_dag(cpdag):
    """
    biased but fast dag sampling from a cpdag
    from Ghassami 2018
    """

    def has_undirected_triple(G, i, j, k):
        for a,b in combinations([i, j, k], 2):
            if G[a,b] == -1:
                return True
        return False

    def has_cycle_triple(G, i, j, k):
        if G[i, j] == 1 and G[j, k] == 1 and G[k, i] == 1:
            return True
        if G[j, i] == 1 and G[i, k] == 1 and G[k, j] == 1:
            return True
        return False

    def has_new_collider_triple(G, already_dir, i, j, k):
        #returns true if the triple has a collider
        for a, b, c in itertools.permutations([i, j, k], 3):
            #if we have a collider
            if G[a][b] == 1 and G[c][b] == 1 and not (G[a][c] != 0 or G[c][a] != 0 or already_dir[a][c] == 1 or already_dir[c][a] == 1):
                #that was not already present
                if not (already_dir[a][b] == 1 and already_dir[c][b] == 1):
                    return True
        return False

    def invalid_dag(G, already_dir, rand_order):
        for i_index in range(n):
            i = rand_order[i_index]
            for j_index in range(i_index+1, n):
                j = rand_order[j_index]
                for k_index in range(j_index+1, n):
                    k = rand_order[k_index]
                    triple_sat = not (has_cycle_triple(G, i, j, k) or has_new_collider_triple(G, already_dir, i, j, k) or has_undirected_triple(G, i, j, k))
                    if not triple_sat:
                        return True
        return False


    n = cpdag.shape[0]
    rand_order = np.random.permutation(n).tolist()

    #first convert cpdag to fully directed format after removing all directed edges
    G = np.minimum(cpdag, 0)
    already_dir = np.maximum(cpdag, 0) #save already directed edges to put back at the end
    #still -1 for undirected edge

    #keep going until a run where we finish with bad_dag = False
    bad_dag = True
    while (bad_dag):
        bad_dag = False
        for i_index in range(n):
            i = rand_order[i_index]
            for j_index in range(i_index+1, n):
                j = rand_order[j_index]
                for k_index in range(j_index+1, n):
                    k = rand_order[k_index]
                    #keep randomly orienting the edges until the triple conditions satisfied
                    #if any of the conditions are true, triple_sat is false
                    triple_sat = not (has_cycle_triple(G, i, j, k) or has_new_collider_triple(G, already_dir, i, j, k) or has_undirected_triple(G, i, j, k))
                    #check triple_sat to start: may not need orient
                    while not triple_sat:
                        for a,b in combinations([i, j, k], 2):
                            #in each case check if there is any edge to orient
                            if (G[a][b] != 0 or G[b][a] != 0):
                                if np.random.binomial(1, 0.5):
                                    G[a][b] = 1
                                    G[b][a] = 0
                                else:
                                    G[a][b] = 0
                                    G[b][a] = 1
                        triple_sat = not (has_cycle_triple(G, i, j, k) or has_new_collider_triple(G, already_dir, i, j, k) or has_undirected_triple(G, i, j, k))

        #now go through everything to check whether we have a valid dag
        bad_dag = invalid_dag(G, already_dir, rand_order)

    return G + already_dir

def uniform_sample_dag_plural(cpdag, num_samples, exact=False):
    """
    Samples multiple DAGs iid uniformly from the MEC
    """
    dags = []
    if exact == False:
        for _ in range(num_samples):
            dags.append(fast_sample_dag(cpdag))
        return dags
    if exact:
        all_dags = enumerate_dags(cpdag)
        return all_dags[np.random.randint(len(all_dags),size=num_samples)]


def enumerate_dags(cpdag):
    """
    a makeshift way of enumerating all days in an MEC by using the sampler
    input:
    matrix cpdag
    output:
    list of dags
    """
    dags = []
    total_dags =size_he(cpdag.copy())

    while len(dags) < total_dags:

        new_dag = fast_sample_dag(cpdag)
        in_list = False
        for dag in dags:
            if np.array_equal(dag, new_dag):
                in_list = True
        if not in_list:
            dags.append(new_dag)

    return dags

if __name__ == '__main__':
    np.random.seed(42)

    for _ in range(20):
        dag1 = main.generate_ER(20, 0.25)
        cpdag1 = main.cpdag_from_dag_observational(dag1)
        print("new round")
        time0=time.time()
        enumerate_dags(cpdag1)
        print(time.time()-time0)
        time0 = time.time()
        enumerate_dags(cpdag1)
        print(time.time()-time0)
        time0 = time.time()
        enumerate_dags(cpdag1)
        print(time.time()-time0)





