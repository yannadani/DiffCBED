import numpy as np
import csv

"""
Helper file for the experiments on GeneNetWeaver simulations
"""
def load_true_dag(nnodes, f):
    """
    input: numberof nodes and file name
    """
    dag = np.zeros((nnodes, nnodes))
    f_file = open(f)

    read_tsv = csv.reader(f_file, delimiter="\t")

    obs = []
    nsamples_obs = 0
    for row in read_tsv:
        i = int(row[0][1:]) - 1
        j = int(row[1][1:]) - 1
        dag[i][j] = 1
    f_file.close()
    return dag










