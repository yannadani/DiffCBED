import networkx as nx
import causaldag as cd
from scipy import special
import numpy as np

try:
    from jax import vmap
    import jax.numpy as jnp
except RuntimeError:
    vmap = None
    jnp = None


def binary_entropy(probs):
    probs = probs.copy()
    probs[probs < 0] = 0
    probs[probs > 1] = 1
    return special.entr(probs) - special.xlog1py(1 - probs, -probs)


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
            variances[node] = error_mle_variance
            amat[node_parents, node] = node_mle_coefficents
    return cd.GaussDAG.from_amat(amat, variances=variances)


def adj_mat_to_vec_single_jax(matrix_full):
    num_nodes = np.shape(matrix_full)[-1]
    upper_tria = jnp.triu_indices(n=num_nodes, k=1)
    lower_tria = jnp.tril_indices(n=num_nodes, k=-1)

    upper_tria_el = matrix_full.at[upper_tria].get()
    lower_tria_el = matrix_full.at[lower_tria].get()

    return jnp.concatenate([upper_tria_el, lower_tria_el], axis=-1)


def adj_mat_to_vec_jax(matrix_full):
    return vmap(adj_mat_to_vec_single_jax, 0, 0)(jnp.array(matrix_full))


def adj_mat_to_vec(matrix_full):
    num_nodes = matrix_full.shape[-1]
    for xx in range(num_nodes):
        matrix_full[:, xx] = np.roll(matrix_full[:, xx], -xx, axis=-1)
    matrix = np.reshape(
        matrix_full[..., 1:], (matrix_full.shape[0], num_nodes * (num_nodes - 1))
    )
    return matrix

def plot_eigs(model, env, args, filename=f'eigs/eigs_2d_res.png'):

    from itertools import product
    from strategies import PriorContrastriveEstimator, PolicyOptPCE
    import pandas as pd
    import pickle
    import io
    import matplotlib.pyplot as plt

    x = jnp.arange(-10, 10+1, 1)

    values = jnp.float32([*product(x, repeat=2)])
    #designs = jnp.stack([nodes, values], -1)

    #nodes = torch.zeros_like(values).long()
    #nodes = F.one_hot(nodes, 2)

    # strategy = PolicyOptPCE(model, env, args)

    estimator = PriorContrastriveEstimator(model, args)

    # valid_interventions = list(range(args.num_nodes))
    #interventions, aux = strategy.acquire(valid_interventions, 0)

    fig, ax = plt.subplots(
        ncols=args.num_nodes,
        nrows=args.num_nodes,
        figsize=(args.num_nodes*5, args.num_nodes*5))

    for i in range(args.num_nodes):
        for j in range(args.num_nodes):
            # 2D designs
            nodes = jnp.int32(jnp.zeros_like(values))
            nodes = nodes.at[:, 0].set(i)
            nodes = nodes.at[:, 1].set(j)

            eig_values = estimator(nodes=nodes, values=values)
            designs_df = pd.DataFrame(
                {
                    "design 1": values[:, 0],
                    "design 2": values[:, 1],
                    "MI": eig_values,
                }
            )

            df_pivot = designs_df.pivot(index="design 1", columns="design 2", values="MI")
            X = df_pivot.columns.values
            Y = df_pivot.index.values
            Z = df_pivot.values
            Xi, Yi = np.meshgrid(X, Y)

            cp = ax[i][j].contourf(Yi, Xi, Z, alpha=0.7, cmap=plt.cm.jet)
            fig.colorbar(cp, ax=ax[i][j]) # Add a colorbar to a plot
            #ax[i][j].set_title(f'lower bound of EIG when intervening on nodes {i} and {j}')
            ax[i][j].set_xlabel(f'node {i} value')
            ax[i][j].set_ylabel(f'node {j} value')
            ax[i][j].set(ylim=[-10, 10])
            ax[i][j].set(xlim=[-10, 10])

    fig.savefig(filename)
    # import pdb; pdb.set_trace()
    # buf = io.BytesIO()
    # pickle.dump(fig, buf)

    # _n = jnp.int32(aux['nodes']).reshape(1, -1)
    # _v = jnp.float32(aux['values']).reshape(1, -1)
    # buf.seek(0)
    # fig2 = pickle.load(buf)
    # for i in range(_v.shape[0]):
    #     _ = fig2.axes[_n[i, 0]*6+_n[i, 1]].scatter(_v[i, 0], _v[i, 1], color='black', alpha=0.2)
    #     fig2.savefig(filename)