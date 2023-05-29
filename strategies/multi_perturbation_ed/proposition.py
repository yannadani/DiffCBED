import numpy as np
import main
import mec_size

def is_subdag(dag, cpdag):
    """
    check if the dag is consistent with a cpdag
    """
    outcome = True
    for i in range(dag.shape[0]):
        for j in range(dag.shape[0]):
            if dag[i,j] == 1 and cpdag[i,j] == 0:
                return False
            if dag[i,j] == 0 and cpdag[i,j] == 1:
                return False
    return outcome

def obj(cpdag, all_dags, len_all_dags, I):
    """
    computes the infinite sample objective value on intervention I for
    set of dags all_dags. 
    """
    f_I = 0
    #average across all possible true dags
    for true_dag in all_dags:
        #ess tracks how many other DAGs in all DAGs are compatible with the I-ess graph of true_dag
        ess = 0
        new_cpdag = main.orient_from_intervention(true_dag, cpdag.copy(), I, is_tree=False)
        for dag in all_dags:
            #check whether the dag is contained in the codag
            if is_subdag(dag, new_cpdag):
                ess+=1
        if ess == 0:
            print(true_dag)
            print(new_cpdag)
            raise Exception
        f_I += np.log(ess)
    return -f_I

dag = np.asarray([[0, 1, 0, 0,  0, 0],
   [0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0],
   [1, 1, 0, 1, 0, 1],
   [0, 0, 1, 0, 0, 0]])
                    
p=4
cpdag = main.cpdag_from_dag_observational(dag)
all_dags = [dag] 

dag2 = np.asarray([[0, 1, 0, 0,  0, 0],
   [0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 1, 0],
   [1, 1, 0, 0, 0, 1],
   [0, 0, 1, 0, 0, 0]])

dag3 = np.asarray([[0, 0, 0, 0,  0, 0],
   [1, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0],
   [1, 1, 0, 1, 0, 1],
   [0, 0, 1, 0, 0, 0]])

all_dags = all_dags + [dag2, dag3]

len_all_dags = len(all_dags)
print(all_dags)
I2 = [np.arange(1, p)]
I1 = [np.arange(1, p-1)]

f_I1 = obj(cpdag, all_dags, len_all_dags, I1)

f_I2 = obj(cpdag, all_dags, len_all_dags, I2)

I2v = [np.arange(p)]
I1v = [np.arange(p-1)]
f_I1v = obj(cpdag, all_dags, len_all_dags, I1v)

f_I2v = obj(cpdag, all_dags, len_all_dags, I2v)
print("f_I1v: " + str(f_I1v))
print("f_I1: " + str(f_I1))
print("f_I2v: " + str(f_I2v))
print("f_I2: " + str(f_I2))

print("f(I1+v) - f(I1)")
print(f_I1v-f_I1)
print("f(I2+v) - f(I2)")
print(f_I2v-f_I2)
if f_I1v-f_I1 < f_I2v-f_I2: 
    print("not submodular!")
    raise Exception




