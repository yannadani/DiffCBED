import time
from collections import defaultdict
from functools import partial
from itertools import product

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import tensorflow_probability as tfp
from bayes_opt import BayesianOptimization
from jax import value_and_grad
from jax.example_libraries import optimizers

from envs.samplers import Constant

from .value_acquisition_strategy import (
    BOValueAcquisitionStrategy,
    FixedValueAcquisitionStrategy,
    GridValueAcquisitionStrategy,
    LinspacePlot,
    MarginalDistValueAcquisitionStrategy,
    RandomValueAcquisitionStrategy,
)

value_acquistion_strategies = {
    "gp-ucb": BOValueAcquisitionStrategy,
    "sample-dist": MarginalDistValueAcquisitionStrategy,
    "fixed": FixedValueAcquisitionStrategy,
    "random": RandomValueAcquisitionStrategy,
    "grid": GridValueAcquisitionStrategy,
    "linspace-plot": LinspacePlot,
}

# This is needed to stop logging of warning
# "WARNING:root:The use of `check_types` is deprecated and does not have any effect."
import logging

logger = logging.getLogger("root")


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())


EPSILON = 1e-8


class AcquisitionStrategy(object):
    def __init__(self, model, env, args):
        self.model = model
        self.args = args
        self.num_samples = args.num_samples
        self.batch_size = args.batch_size
        self.env = env
        self.value_strategy = args.value_strategy

    def acquire(self, nodes, iteration):
        strategy = self.get_value_strategy(nodes)
        strategy(self._score_for_value, n_iters=self.args.num_intervention_values)

        selected_interventions = defaultdict(list)

        selected_interventions[strategy.max_j].extend(
            [Constant(strategy.max_x)] * self.batch_size
        )

        scores = None
        return selected_interventions, scores

    def get_value_strategy(self, nodes):
        strategy = value_acquistion_strategies[self.value_strategy](
            nodes=nodes, args=self.args
        )
        return strategy


def scatter(input, dim, index, src, reduce=None):
    idx = jnp.meshgrid(*(jnp.arange(n) for n in input.shape), sparse=True)
    idx[dim] = index
    return getattr(input.at[tuple(idx)], reduce or "set")(src)


class SubsetOperator:
    def __init__(self, k, num_designs, num_nodes, hard=False):
        self.hard = hard
        self.k = k
        self.num_designs = num_designs
        self.num_nodes = num_nodes

    def sample(self, rng_seq, B, scores, tau=1.0):
        gumbel_noise = jax.random.gumbel(
            next(rng_seq), (B, self.num_designs, self.num_nodes)
        )

        # m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        # g = m.sample()
        scores = scores + gumbel_noise

        # y = gumbel_noise + logits
        # node_samples = jax.nn.softmax(y / temperature, -1)
        # nodes2 = jax.nn.one_hot(node_samples.argmax(-1), self.num_nodes)

        # continuous top k
        khot = jnp.zeros_like(scores)
        onehot_approx = jnp.zeros_like(scores)
        for i in range(self.k):
            khot_mask = jnp.clip(1.0 - onehot_approx, EPSILON)
            scores = scores + jnp.log(khot_mask)
            onehot_approx = jax.nn.softmax(scores / tau)
            khot = khot + onehot_approx

        if self.hard:
            # will do straight through estimation if training
            khot_hard = jnp.zeros_like(khot)
            val, ind = jax.lax.top_k(khot, self.k)

            for i in range(khot_hard.shape[0]):
                for j in range(khot_hard.shape[1]):
                    khot_hard = khot_hard.at[i, j, ind[i][j]].set(1)

            # khot_hard = scatter(khot_hard, 2, ind, 1)
            res = khot_hard - jax.lax.stop_gradient(khot) + khot
        else:
            res = khot

        return res


class Policy(object):
    def __init__(self, num_nodes, num_targets, num_designs, node_range):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_designs = num_designs
        self.num_targets = num_targets
        self.node_range = node_range

        if num_targets > 0:
            self.subset = SubsetOperator(
                k=num_targets, num_designs=num_designs, num_nodes=num_nodes, hard=True
            )

    def __call__(self, key, batch_size, temperature=1.0):
        rng_seq = hk.PRNGSequence(key)

        # mean = hk.get_parameter('mean', [self.num_designs], init=jnp.zeros)
        stddev_init = hk.initializers.Constant(self.node_range[1])
        # stddev = hk.get_parameter('stddev', [self.num_designs], init=stddev_init)
        logits = hk.get_parameter(
            "logits", [self.num_designs, self.num_nodes], init=jnp.zeros
        )

        # binary = False

        # if binary:
        #     dist = tfp.substrates.jax.distributions.RelaxedOneHotCategorical(temperature=temperature, logits=logits)
        #     node_samples = dist.sample(seed=next(rng_seq), sample_shape=(batch_size,))
        #     nodes = jax.nn.one_hot(node_samples.argmax(-1), self.num_nodes)
        # else:
        #     dist = tfp.substrates.jax.distributions.RelaxedBernoulli(temperature=temperature, logits=logits)
        #     node_samples = dist.sample(seed=next(rng_seq), sample_shape=(batch_size,))
        #     nodes = (node_samples > .5)

        # nodes = node_samples + jax.lax.stop_gradient(nodes - node_samples)

        # # Gaussian for values
        # values_dist = distrax.MultivariateNormalDiag(
        #    loc=mean, scale_diag=jax.nn.softplus(stddev) + 1e-1)
        # values = values_dist._sample_n(key=next(rng_seq), n=batch_size)

        values = hk.get_parameter(
            "values",
            [self.num_designs, self.num_nodes],
            init=hk.initializers.RandomUniform(self.node_range[0], self.node_range[1]),
        )
        # # values = hk.get_parameter('values', [self.num_designs], init=jnp.zeros)

        # straight through
        if self.num_targets > 0:
            node_samples_hard_grad = self.subset.sample(
                rng_seq, batch_size, logits, tau=temperature
            )
        else:
            dist = tfp.substrates.jax.distributions.RelaxedBernoulli(
                temperature=temperature, logits=logits
            )
            node_samples = dist.sample(seed=next(rng_seq), sample_shape=(batch_size,))
            node_samples_hard = node_samples.round()
            node_samples_hard_grad = (
                node_samples - jax.lax.stop_gradient(node_samples) + node_samples_hard
            )
        # import pdb; pdb.set_trace()

        # # Gumbel-Softmax for nodes
        # gumbel_noise = jax.random.gumbel(next(rng_seq), (batch_size, self.num_designs, self.num_nodes))
        # y = gumbel_noise + logits
        # node_samples = jax.nn.softmax(y / temperature, -1)
        # nodes2 = jax.nn.one_hot(node_samples.argmax(-1), self.num_nodes)

        return node_samples_hard_grad, values


class PolicyWithFixedValue(object):
    def __init__(self, num_nodes, num_targets, num_designs, node_range, fixed_value):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_designs = num_designs
        self.num_targets = num_targets
        self.node_range = node_range
        self.fixed_value = fixed_value

        if num_targets > 0:
            self.subset = SubsetOperator(
                k=num_targets, num_designs=num_designs, num_nodes=num_nodes, hard=True
            )

    def __call__(self, key, batch_size, temperature=1.0):
        rng_seq = hk.PRNGSequence(key)
        logits = hk.get_parameter(
            "logits", [self.num_designs, self.num_nodes], init=jnp.zeros
        )

        values = jnp.ones([self.num_designs, self.num_nodes]) * self.fixed_value

        # straight through
        if self.num_targets > 0:
            node_samples_hard_grad = self.subset.sample(
                rng_seq, batch_size, logits, tau=temperature
            )
        else:
            dist = tfp.substrates.jax.distributions.RelaxedBernoulli(
                temperature=temperature, logits=logits
            )
            node_samples = dist.sample(seed=next(rng_seq), sample_shape=(batch_size,))
            node_samples_hard = node_samples.round()
            node_samples_hard_grad = (
                node_samples - jax.lax.stop_gradient(node_samples) + node_samples_hard
            )

        return node_samples_hard_grad, values


class GradientBasedPolicyOptimizer(object):
    def __init__(self, args):
        self.args = args

    def optimize(self, subk, f, policy):
        # negate utility to make it a loss
        def loss(params, key1, key2, temperature=5.0, val_range=[-10, 10]):
            nodes, values = policy.apply(
                params=params,
                rng=key1,
                key=key2,
                batch_size=32,
                temperature=temperature,
            )

            if (
                "fixed_value" in self.args.strategy
                or self.args.value_strategy == "fixed"
            ):
                values = values[None].repeat(nodes.shape[0], 0)
            else:
                soft_tanh = (
                    lambda x: jax.nn.hard_sigmoid(
                        (x - val_range[0]) / (val_range[1] - val_range[0])
                    )
                    * (val_range[1] - val_range[0])
                    + val_range[0]
                )
                values = soft_tanh(values)[None].repeat(nodes.shape[0], 0)

            eig = f(nodes=nodes, values=values)
            idx = (-eig).argsort()[:10]
            aux = {"nodes": nodes, "values": values, "eig": eig, "idx": idx}
            return -eig.mean(), aux

        rng_seq = hk.PRNGSequence(subk)
        params = policy.init(next(rng_seq), key=next(rng_seq), batch_size=1)
        # x = jnp.arange(self.args.node_range[0], self.args.node_range[1], 1)

        # values = jnp.float32([*product(x, repeat=2)])

        epochs = self.args.opt_epochs

        optimizer = optax.adam(self.args.opt_lr)
        opt_state = optimizer.init(params)

        # TODO: keep the best
        cem = False
        best_params = None
        best_eig = -1
        if self.args.temperature_type == "anneal":
            temp_max, temp_min, temp_steps = 5, 0.5, 80
        else:
            temp_max, temp_min, temp_steps = (
                self.args.temperature,
                self.args.temperature,
                1,
            )
        history = []
        for epoch in range(epochs):
            t0 = time.time()

            temperature = temp_max - (temp_max - temp_min) / temp_steps * min(
                epoch, temp_steps
            )

            (eig, aux), grads = value_and_grad(loss, has_aux=True)(
                params, next(rng_seq), next(rng_seq), temperature, self.args.node_range
            )

            history.append(aux)

            if "values" in grads["~"]:
                grads_mag = (grads["~"]["logits"] ** 2).mean() + (
                    grads["~"]["values"] ** 2
                ).mean()

                t1 = time.time()
                print(
                    f"{t1-t0}, eig:{eig}, grads_mag:{grads_mag}, var:{grads['~']['logits'].var():.6f}"
                )
            else:
                grads_mag = (grads["~"]["logits"] ** 2).mean()

                t1 = time.time()
                print(
                    f"{t1-t0}, eig:{eig}, grads_mag:{grads_mag}, var:{grads['~']['logits'].var():.6f}"
                )

            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # (Pdb) p values.shape
            # (5, 10)
            # (Pdb) p nodes.shape
            # (5, 10)

            # descending ordering of eig and keep top 10

            # idx = aux['idx']

            # params['~']['mean'] = aux['values'].mean(0)
            # params['~']['stddev'] = aux['values'].std(0)

            # params['~']['logits'] = params['~']['logits'].at[idx].add(0.01)

            # Cross Entropy Method for the values
            if cem:
                idx = aux["idx"]

                params["~"]["mean"] = aux["values"][idx].mean(0)
                params["~"]["stddev"] = aux["values"][idx].std(0)

                params["~"]["mean"] = jnp.clip(
                    params["~"]["mean"],
                    self.args.node_range[0],
                    self.args.node_range[1],
                )
                params["~"]["stddev"] = jnp.clip(params["~"]["stddev"], 0, 20)

            # import pdb; pdb.set_trace()

            if eig < best_eig or best_params is None:
                best_eig = eig
                best_params = params.copy()

            # assert (params['~']['logits']**2).mean() > 0
            # assert (params['~']['mean']**2).mean() > 0
            # assert (params['~']['stddev']**2).mean() > 0

        print(f"Returning params scoring best eig {best_eig}")

        return best_params, {"history": history}


class GridOptimizer(object):
    def __init__(self, args):
        self.args = args

    def optimize(self, subk, f, policy):
        # negate utility to make it a loss
        def loss(params, key1, key2, temperature=5.0, val_range=[-10, 10]):
            nodes, values = policy.apply(
                params=params,
                rng=key1,
                key=key2,
                batch_size=128,
                temperature=temperature,
            )
            values = values.clip(val_range[0], val_range[1])
            eig = f(nodes=nodes, values=values)
            idx = (-eig).argsort()[:10]
            aux = {"nodes": nodes, "values": values, "eig": eig, "idx": idx}
            return -eig.mean(), aux

        rng_seq = hk.PRNGSequence(subk)
        params = policy.init(next(rng_seq), key=next(rng_seq), batch_size=1)

        x = jnp.arange(self.args.node_range[0], self.args.node_range[1], 1)

        values = jnp.float32([*product(x, repeat=2)])

        best_params = {}
        best_eig = 0
        for i in range(self.args.num_nodes):
            for j in range(self.args.num_nodes):
                nodes = jnp.int32(jnp.zeros_like(values))
                nodes = nodes.at[:, 0].set(i)
                nodes = nodes.at[:, 1].set(j)

                eig = f(nodes=nodes, values=values)
                designs_df = pd.DataFrame(
                    {
                        "design 1": values[:, 0],
                        "design 2": values[:, 1],
                        "MI": eig,
                    }
                )

                df_pivot = designs_df.pivot(
                    index="design 1", columns="design 2", values="MI"
                )
                X = df_pivot.columns.values
                Y = df_pivot.index.values
                Z = df_pivot.values

                _i, _j = jnp.unravel_index(Z.argmax(), Z.shape)

                if Z[_i][_j] > best_eig:
                    best_eig = Z[_i][_j]
                    best_params["nodes"] = [i, j]
                    best_params["values"] = [X[_i], Y[_j]]
                    print(
                        f"selecting best {best_params['nodes']} {best_params['values']} {best_eig}"
                    )

        return best_params, {}


class GradientBasedOptimizer(object):
    def __init__(self, args):
        self.args = args

    def optimize(self, subk, f, n_params):
        vals = jax.random.normal(key=subk, shape=(1, n_params))

        # negate utility to make it a loss
        def loss(*args):
            return -f(*args)

        step_size = self.args.opt_lr
        epochs = self.args.opt_epochs
        opt_init, opt_update, get_params = optimizers.adam(step_size)
        opt_state = opt_init(vals)
        for _ in range(epochs):
            t0 = time.time()
            eig, grads = value_and_grad(loss)(vals)
            opt_state = opt_update(0, grads, opt_state)
            vals = get_params(opt_state)
            t1 = time.time()
            print(f"{t1-t0},{eig},{(grads**2).mean():.6f}")
        return vals


class MultiNodeGradientBasedOptimizer(object):
    def __init__(self, args):
        self.args = args

    def optimize(self, subk, f, n_params):
        nodes = jnp.arange(self.args.num_nodes)[:, None]
        values = jax.random.normal(key=subk, shape=(self.args.num_nodes, 1))  # Values
        designs = jnp.stack([nodes, values], axis=-1)[:, None]

        step_size = self.args.opt_lr
        epochs = self.args.opt_epochs

        # negate utility to make it a loss
        def loss(*args):
            return -f(*args)

        opts = [optimizers.adam(step_size) for _ in range(self.args.num_nodes)]
        opt_states = [opts[i][0](designs[i]) for i in range(self.args.num_nodes)]

        grad_fn = jax.vmap(value_and_grad(loss))
        scores = []
        values = []
        for _ in range(epochs):
            t0 = time.time()
            score, grads = grad_fn(designs)
            opt_states = [
                opts[i][1](0, grads[i], opt_states[i])
                for i in range(self.args.num_nodes)
            ]
            designs = jnp.stack(
                [opts[i][2](opt_states[i]) for i in range(self.args.num_nodes)]
            )
            t1 = time.time()
            scores.append(score)
            values.append(designs[..., 0, 0, 1])
            print(f"{t1-t0}", score)

        return jnp.array(scores), jnp.array(values)


class BayesianOptimizer(object):
    def __init__(self, args):
        self.args = args
        self.bounds = (-100, 100)

    def optimize(self, subk, f, n_params):
        def utility(**args):
            values = jnp.array(list(args.values()))[None, :]
            return f(values)

        optimizer = BayesianOptimization(
            f=utility,
            pbounds=dict([(f"x_{i}", self.bounds) for i in range(n_params)]),
            verbose=2,
            random_state=self.args.seed,
        )

        optimizer.maximize(
            n_iter=self.args.opt_epochs, acq="ucb", kappa=2.5, init_points=1, xi=0.0
        )

        return jnp.array(list(optimizer.max["params"].values()))[None]


class MultiNodeBayesianOptimizer(object):
    def __init__(self, args):
        self.args = args
        self.bounds = (-100, 100)

    def optimize(self, subk, f, n_params):
        def utility(node, value):
            design = jnp.array([node, value])[None, None]
            return f(design)

        optimizers = [
            BayesianOptimization(
                f=partial(utility, node=node),
                pbounds={"value": self.bounds},
                verbose=2,
                random_state=self.args.seed,
            )
            for node in range(self.args.num_nodes)
        ]

        scores = []
        values = []
        for optimizer in optimizers:
            optimizer.maximize(
                n_iter=self.args.opt_epochs, acq="ucb", kappa=2.5, init_points=1, xi=0.0
            )
            scores.append([r["target"] for r in optimizers[0].res])
            values.append([r["params"]["value"] for r in optimizers[0].res])

        # nodes x iters
        return jnp.array(scores).T, jnp.array(values).T


def softmax(vals, temp=1.0):
    """Batch softmax
    Args:
        vals (np.ndarray): S x A. Applied row-wise
        t (float, optional): Defaults to 1.. Temperature parameter
    Returns:
        np.ndarray: S x A
    """
    return np.exp(
        (1.0 / temp) * vals - logsumexp((1.0 / temp) * vals, axis=1, keepdims=True)
    )


class PolicyOptimization(object):
    def __init__(self, model, env, args, optimizer, utility):
        self.model = model
        self.args = args
        self.num_samples = args.num_samples
        self.batch_size = args.batch_size
        self.env = env
        self.value_strategy = args.value_strategy
        self.key = jax.random.PRNGKey(self.args.seed)
        self.optimizer = optimizer(args)
        self.utility = utility(model, args)
        self.policy = hk.transform(
            Policy(
                self.args.num_nodes,
                self.args.num_targets,
                self.batch_size,
                self.args.node_range,
            )
        )

    def acquire(self, nodes, iteration):
        self.key, subk = jax.random.split(self.key)

        def utility(nodes, values):
            return self.utility(nodes=nodes, values=values, onehot=True)

        self.key, subk = jax.random.split(self.key)
        policy_params, aux = self.optimizer.optimize(subk, utility, self.policy)

        # argmax from history
        best = 0
        if "history" in aux:
            for h in range(len(aux["history"])):
                eigs = aux["history"][h]["eig"]
                idx = eigs.argmax()
                if best == 0 or eigs[idx] > best:
                    best = eigs[idx]
                    nodes = aux["history"][h]["nodes"][idx]
                    values = aux["history"][h]["values"][idx]
        else:
            nodes = jnp.int32(policy_params["nodes"])
            values = jnp.float32(policy_params["values"])

        return {"nodes": nodes, "values": values}, aux


class PolicyOptimizationWithFixedValue(PolicyOptimization):
    def __init__(self, model, env, args, optimizer, utility):
        super().__init__(model, env, args, optimizer, utility)
        self.policy = hk.transform(
            PolicyWithFixedValue(
                self.args.num_nodes,
                self.args.num_targets,
                self.batch_size,
                self.args.node_range,
                self.args.intervention_value,
            )
        )


class RandomNodeOptimizeValue(object):
    def __init__(self, model, env, args, optimizer, utility):
        self.model = model
        self.args = args
        self.num_samples = args.num_samples
        self.batch_size = args.batch_size
        self.env = env
        self.value_strategy = args.value_strategy
        self.key = jax.random.PRNGKey(self.args.seed)
        self.optimizer = optimizer(args)
        self.utility = utility(model, args)

    def acquire(self, nodes, iteration):
        self.key, subk = jax.random.split(self.key)
        nodes = jnp.int32(
            jax.random.permutation(
                subk, jnp.arange(self.args.num_nodes), independent=True
            )[: self.batch_size]
        )[None]

        def utility(values):
            designs = jnp.stack([nodes, values], -1)
            return self.utility(designs).mean()

        self.key, subk = jax.random.split(self.key)
        values = self.optimizer.optimize(subk, utility, self.batch_size)

        selected_interventions = defaultdict(list)
        for i in range(self.batch_size):
            selected_interventions[nodes[0, i].item()].extend(
                [Constant(values[0, i].item())]
            )

        return selected_interventions, None


class SoftTopK(object):
    def __init__(self, model, env, args, optimizer, utility):
        self.model = model
        self.args = args
        self.num_samples = args.num_samples
        self.batch_size = args.batch_size
        self.env = env
        self.key = jax.random.PRNGKey(self.args.seed)
        self.optimizer = optimizer(args)
        self.utility = utility(model, args)

    def acquire(self, nodes, iteration):
        def utility(designs):
            return self.utility(designs).mean()

        self.key, subk = jax.random.split(self.key)
        scores, values = self.optimizer.optimize(subk, utility, self.batch_size)

        # soft selection
        probs = (
            jnp.exp(scores / self.args.bald_temperature)
            / jnp.exp(scores / self.args.bald_temperature).sum()
        )

        assert (
            self.batch_size < self.args.num_nodes
        ), "Batch size need to be smaller than the number of nodes"

        self.key, subk = jax.random.split(self.key)
        interventions = jax.random.choice(
            key=subk,
            a=len(probs.flatten()),
            p=probs.flatten(),
            replace=False,
            shape=(1, self.batch_size),
        )
        value_ids, node_ids = jnp.unravel_index(interventions[0], shape=probs.shape)

        selected_interventions = defaultdict(list)
        for value_id, node in zip(value_ids, node_ids):
            selected_interventions[nodes[node]].append(
                Constant(values[value_id][nodes[node]])
            )

        return selected_interventions, None
