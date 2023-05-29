from __future__ import division  # in case python2 is used

import csv
import operator as op
import os
import random
import shutil
import uuid
from pathlib import Path

import causaldag as cd
import networkx as nx
import numpy as np

# import config
import pandas as pd
from networkx.utils import powerlaw_sequence

# from sksparse.cholmod import cholesky  # this has to be used instead of scipy's because it doesn't permute the matrix
from scipy import sparse
from scipy.special import logsumexp

import tqdm

def bernoulli(p):
    return np.random.binomial(1, p)


def RAND_RANGE():
    return np.random.uniform(0.25, 1) * (-1 if bernoulli(0.5) else 1)


def run_gies_boot(data, n_boot, group_interventions, maintain_int_dist):
    uid = str(uuid.uuid4())
    tmp_path = Path("tmp/") / uid

    os.makedirs(tmp_path, exist_ok=True)
    dags_path = os.path.join(tmp_path, "dags/")
    os.makedirs(dags_path, exist_ok=True)

    assert (
        data.nodes.dtype == bool
    ), "Please input boolean mask for interventional samples"
    interventions = np.array(data.nodes)

    is_single_target = (interventions.sum((-1)) <= 1).sum() == interventions.shape[0]
    idx = np.array(range(len(interventions)))
    if is_single_target:
        if group_interventions:
            idx = np.argsort(interventions)
    # order samples by interventions to group similar interventions together
    interventions = interventions[idx]
    for i in tqdm.tqdm(range(n_boot)):
        data_indices, unique_targets, target_indices = get_bootstrap_indices(
            interventions, is_single_target, maintain_int_dist=maintain_int_dist
        )
        np.savetxt(tmp_path / "samples.csv", data.samples[data_indices], delimiter=" ")
        with open(tmp_path / "unique_targets.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(unique_targets)
        np.array(target_indices).tofile(
            tmp_path / "target_indices.csv", sep="\n", format="%d"
        )
        open(tmp_path / "target_indices.csv", "a").write("\n")
        if not os.path.exists(dags_path):
            os.mkdir(dags_path)
        rfile = os.path.join("models", "dag_bootstrap_lib", "run_gies.r")
        r_command = "Rscript {} {} {} {} {} {}".format(
            rfile,
            str(tmp_path / "samples.csv"),
            str(tmp_path / "unique_targets.csv"),
            str(tmp_path / "target_indices.csv"),
            dags_path,
            i,
        )

        os.system(r_command)
    return tmp_path


def _write_data(data, samples_path, interventions_path):
    """
    Helper function to write interventional data to files so that it can be used by R
    """
    # clear current data
    open(samples_path, "w").close()
    open(interventions_path, "w").close()

    iv_nodes = []
    for iv_node, samples in data.items():
        with open(samples_path, "ab") as f:
            np.savetxt(f, samples)
        iv_nodes.extend([iv_node + 1 if iv_node != -1 else -1] * len(samples))
    pd.Series(iv_nodes).to_csv(interventions_path, index=False)


def generate_DAG(p, m=4, prob=0.0, type_="config_model"):
    if type_ == "config_model":
        z = [int(e) for e in powerlaw_sequence(p)]
        if np.sum(z) % 2 != 0:
            z[0] += 1
        G = nx.configuration_model(z)
    elif type_ == "barabasi":
        G = nx.barabasi_albert_graph(p, m)
    elif type_ == "small_world":
        G = nx.watts_strogatz_graph(p, m, prob)
    elif type_ == "chain":
        source_node = int(np.ceil(p / 2)) - 1
        arcs = {(i + 1, i) for i in range(source_node)} | {
            (i, i + 1) for i in range(source_node, p - 1)
        }
        print(source_node, arcs)
        return cd.DAG(nodes=set(range(p)), arcs=arcs)
    elif type_ == "chain_one_direction":
        return cd.DAG(nodes=set(range(p)), arcs={(i, i + 1) for i in range(p - 1)})
    else:
        raise Exception("Not a graph type")
    G = nx.Graph(G)
    dag = cd.DAG(nodes=set(range(p)))
    for i, j in G.edges:
        if i != j:
            dag.add_arc(*sorted((i, j)))
    return dag


def get_precision_interventional(gdag, iv_node, iv_variance):
    adj = gdag.weight_mat.copy()
    adj[:, iv_node] = 0
    vars = gdag.variances.copy()
    vars[iv_node] = iv_variance
    id_ = np.identity(adj.shape[0])
    return (id_ - adj) @ np.diag(vars**-1) @ (id_ - adj).T


def get_covariance_interventional(gdag, iv_node, iv_variance):
    adj = gdag.weight_mat.copy()
    adj[:, iv_node] = 0
    vars = gdag.variances.copy()
    vars[iv_node] = iv_variance
    id_ = np.identity(adj.shape[0])

    id_min_adj_inv = np.linalg.inv(id_ - adj)
    return id_min_adj_inv.T @ np.diag(vars) @ id_min_adj_inv


def cross_entropy_interventional(gdag1, gdag2, iv_node, iv_variance):
    precision2 = get_precision_interventional(gdag2, iv_node, iv_variance)
    covariance1 = get_covariance_interventional(gdag1, iv_node, iv_variance)
    p = len(gdag1.nodes)
    kl_term = -p / 2
    kl_term += np.trace(precision2 @ covariance1) / 2
    logdet2 = (
        np.sum(np.log(gdag2.variances))
        - np.log(gdag2.variances[iv_node])
        + np.log(iv_variance)
    )
    logdet1 = (
        np.sum(np.log(gdag1.variances))
        - np.log(gdag1.variances[iv_node])
        + np.log(iv_variance)
    )
    kl_term += (logdet2 - logdet1) / 2
    entropy_term = (np.log(2 * np.pi * np.e) * p + logdet1) / 2

    return -1 * (kl_term + entropy_term)


def _load_dags(tmp_path, delete=True):
    """
    Helper function to load the DAGs generated in R
    """
    adj_mats = []
    dags_path = os.path.join(tmp_path, "dags")
    paths = os.listdir(dags_path)
    for file_path in paths:
        if "score" not in file_path and ".DS_Store" not in file_path:
            adj_mat = pd.read_csv(os.path.join(dags_path, file_path))
            adj_mats.append(adj_mat.values)
    if delete:
        shutil.rmtree(tmp_path)
    return adj_mats, [cd.DAG.from_amat(adj) for adj in adj_mats]


def probability_shrinkage(prob):
    return 2 * min(1 - prob, prob)


def entropy_shrinkage(prob):
    if prob == 0 or prob == 1:
        return 0
    return (prob * np.log(prob) + (1 - prob) * np.log(1 - prob)) / np.log(2)


def get_bootstrap_indices(intervention_mask, is_single_target, maintain_int_dist=True):
    """_summary_
    Args:
        intervention_mask (torch.bool): Intervention indices of all the samples (n x d)
    """
    N = intervention_mask.shape[0]
    unique_targets = []
    data_indices = []
    target_indices = []
    interventions = np.nonzero(intervention_mask)
    observational_indices = np.nonzero(intervention_mask.sum((-1)) == 0)[0]
    unique_targets.append([-1])
    data_indices.extend(
        np.random.choice(
            observational_indices, len(observational_indices), replace=True
        )
    )
    target_indices.extend([1] * len(observational_indices))  # R uses 1 indexing
    if is_single_target:
        unique_interventions = np.unique(interventions[1])
        for k, i in enumerate(unique_interventions):
            unique_targets.append([i + 1])
            curr_indcs = interventions[0][interventions[1] == i]
            data_indices.extend(
                np.random.choice(curr_indcs, len(curr_indcs), replace=True)
            )
            target_indices.extend([k + 2] * len(curr_indcs))
    else:
        interventional_indices = np.nonzero(~(intervention_mask.sum((-1)) == 0))[0]
        all_interventions = np.array(
            [np.nonzero(y)[0] for y in intervention_mask], dtype=object
        )
        if maintain_int_dist:
            unique_interventions_tuple = [
            y for y in set([tuple(x) for x in all_interventions])
        ]
            hash_list_unique_interventions = [
                hash(y) for y in unique_interventions_tuple
            ]
            hash_list_all_interventions = [hash(tuple(y)) for y in all_interventions]
            idx_ = 0
            for k,i in enumerate(unique_interventions_tuple):
                if i == ():
                    continue
                unique_targets.append([element + 1 for element in i])
                equal_indices = np.nonzero(np.array(hash_list_all_interventions) == hash_list_unique_interventions[k])[0]
                selected_indices = np.random.choice(equal_indices, len(equal_indices), replace=True)
                data_indices.extend(selected_indices)
                target_indices.extend([idx_+2]*len(equal_indices))
                idx_+=1
        else:
            selected_indices = np.random.choice(
                interventional_indices, len(interventional_indices), replace=True
            )
            data_indices.extend(selected_indices)

            selected_interventions = all_interventions[selected_indices]
            unique_selected_interventions_tuple = [
                y for y in set([tuple(x) for x in selected_interventions])
            ]
            hash_list_unique_interventions = [
                hash(y) for y in unique_selected_interventions_tuple
            ]
            for i in unique_selected_interventions_tuple:
                unique_targets.append([element + 1 for element in i])
            for k, i in enumerate(selected_interventions):
                hash_code = hash(tuple(i))
                target_indices.extend([hash_list_unique_interventions.index(hash_code) + 2])
    return data_indices, unique_targets, target_indices


# def prec2dag(prec, node_order):
#     p = prec.shape[0]
#
#     # === permute precision matrix into correct order for LDL
#     prec = prec.copy()
#     rev_node_order = list(reversed(node_order))
#     prec = prec[rev_node_order]
#     prec = prec[:, rev_node_order]
#
#     # === perform ldl decomposition and correct for floating point errors
#     factor = cholesky(sparse.csc_matrix(prec))
#     l, d = factor.L_D()
#     l = l.todense()
#     d = d.todense()
#
#     # === permute back
#     inv_rev_node_order = [i for i, j in sorted(enumerate(rev_node_order), key=op.itemgetter(1))]
#     l = l.copy()
#     l = l[inv_rev_node_order]
#     l = l[:, inv_rev_node_order]
#     d = d.copy()
#     d = d[inv_rev_node_order]
#     d = d[:, inv_rev_node_order]
#
#     amat = np.eye(p) - l
#     variances = np.diag(d) ** -1
#
#     return cd.GaussDAG.from_amat(amat, variances=variances)


# def cov2dag(cov_mat, dag):
#     # See formula https://arxiv.org/pdf/1303.3216.pdf pg. 17
#     nodes = dag.nodes
#     p = len(nodes)
#     amat = np.zeros((p, p))
#     variances = np.zeros(p)
#     for node in nodes:
#         node_parents = list(dag.parents[node])
#         if len(node_parents) == 0:
#             variances[node] = cov_mat[node, node]
#         else:
#             S_k_k = cov_mat[node, node]
#             S_k_pa = cov_mat[node, node_parents]
#             S_pa_pa = cov_mat[np.ix_(node_parents, node_parents)]
#             if len(node_parents) > 1:
#                 inv_S_pa_pa = np.linalg.inv(S_pa_pa)
#             else:
#                 inv_S_pa_pa = np.array(1 / S_pa_pa)
#             node_mle_coefficents = S_k_pa.dot(inv_S_pa_pa)
#             error_mle_variance = S_k_k - S_k_pa.dot(inv_S_pa_pa.dot(S_k_pa.T))
#             variances[node] = error_mle_variance
#             amat[node_parents, node] = node_mle_coefficents
#     return cd.GaussDAG.from_amat(amat, variances=variances)


def cov2dag(cov_mat, dag):
    # See formula https://arxiv.org/pdf/1303.3216.pdf pg. 17
    nodes = dag.nodes
    p = len(nodes)
    amat = np.zeros((p, p))
    variances = np.zeros(p)
    for node in nodes:
        node_parents = list(dag.parents[node])
        if len(node_parents) == 0:
            variances[node] = cov_mat[node, node]
        else:
            S_k_k = cov_mat[node, node]
            S_k_pa = cov_mat[node, node_parents]
            S_pa_pa = cov_mat[np.ix_(node_parents, node_parents)]
            if len(node_parents) > 1:
                inv_S_pa_pa = np.linalg.inv(S_pa_pa)
            else:
                inv_S_pa_pa = np.array(1 / S_pa_pa)
            node_mle_coefficents = S_k_pa.dot(inv_S_pa_pa)
            error_mle_variance = S_k_k - S_k_pa.dot(inv_S_pa_pa.dot(S_k_pa.T))
            # clip variances - is this correct?
            variances[node] = (
                np.clip(error_mle_variance, 0, np.abs(error_mle_variance)) + 1e-6
            )
            amat[node_parents, node] = node_mle_coefficents
    return cd.GaussDAG.from_amat(amat, variances=variances)


def dag_posterior(dag_collec, data, intervention_nodes, interventions):
    logpdfs = np.zeros(len(dag_collec))
    for dag_ix, cand_dag in enumerate(dag_collec):
        for iv, samples in data.items():
            if iv == -1:
                logpdfs[dag_ix] += cand_dag.logpdf(samples).sum()
            else:
                iv_ix = intervention_nodes.index(iv)
                logpdfs[dag_ix] += cand_dag.logpdf(
                    samples, interventions={iv: interventions[iv_ix]}
                ).sum()
    return np.exp(logpdfs - logsumexp(logpdfs))


if __name__ == "__main__":
    from collections import namedtuple

    import numpy as np

    Data = namedtuple("data", ["samples", "nodes"])

    num_samples, num_nodes = 5, 4
    samples = np.random.normal(size=(num_samples, num_nodes))
    nodes = np.zeros((num_samples, num_nodes)).astype(bool)
    nodes[2, 1], nodes[3, 3], nodes[3, 0], nodes[4, 0], nodes [4, 3]  = True, True, True, True, True

    print(nodes)

    dags_path_ = run_gies_boot(
        Data(samples=samples, nodes=nodes), n_boot=5, group_interventions=False, maintain_int_dist=
        True
    )
    dags, dags_amat = _load_dags(tmp_path=dags_path_, delete=True)

    print(dags_amat)