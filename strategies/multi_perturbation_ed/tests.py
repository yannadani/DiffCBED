import networkx as nx
from main import *
from mec_size import *
import math

#check the trees being generated are actual trees
#only checks if the skeleton is a tree
#also checks ss_construct
for _ in range(0, 10):
    n = 20
    tree_dag = uniform_random_tree(n)
    tree = nx.from_numpy_matrix(tree_dag)
    assert nx.is_tree(tree), "Expected tree"

    #also check that the separating system separates
    for k in range(1, 4):
        tree_cpdag = cpdag_from_dag_observational(tree_dag)
        tree_ref_cpdag = tree_cpdag.copy()
        ss = ss_construct(n, k)
        #bit of rounding error allowed
        assert objective_given_intervention(tree_cpdag, ss, tree_ref_cpdag) > n-1 - 0.1, "Not a separating system"


#now test counting MECs works
#example from the paper
cpdag1 = np.asarray([[0, -1, -1, 0], [-1, 0, -1, -1], [-1, -1, 0, -1], [0, -1, -1, 0]])
assert mec_size(cpdag1, []) == 10
#try some trees
for n in range(3, 10):
    dag1 = main.uniform_random_tree(n)
    cpdag1 = main.cpdag_from_dag_observational(dag1)
    assert mec_size(cpdag1, []) == n

for n in range(3, 10):
    dag1 = generate_fully_connected(n)
    cpdag1 = main.cpdag_from_dag_observational(dag1)
    assert mec_size(cpdag1, []) == math.factorial(n)

for n in range(3, 10):
    dag1 = generate_chain_dag_no_colliders(n)
    cpdag1 = main.cpdag_from_dag_observational(dag1)
    assert mec_size(cpdag1, []) == n

cpdag = np.asarray(generate_ER(40, 0.1))

#check enumerating from MEC gives dags
mec = enumerate_dags(cpdag, [], exact=True)
for dag in mec:
    g = nx.DiGraph(dag)
    order = list(nx.topological_sort(g))

alt_mec = enumerate_dags(cpdag, [], exact=False)
for g in mec:
    in_mec = False
    for g2 in alt_mec:
        if np.all(g == g2):
            in_mec=True
    assert(in_mec)

print("All tests passed")