from functools import partial

import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp

from .acquisition_strategy import (
    BayesianOptimizer,
    GradientBasedOptimizer,
    GradientBasedPolicyOptimizer,
    GridOptimizer,
    MultiNodeBayesianOptimizer,
    MultiNodeGradientBasedOptimizer,
    PolicyOptimization,
    PolicyOptimizationWithFixedValue,
    RandomNodeOptimizeValue,
    SoftTopK,
)


def policy():
    return 0, 0.0


def logmeanexp(A, axis):
    return logsumexp(A, axis) - jnp.log(A.shape[axis])


class Utility(object):
    def __init__(self, model, args):
        self.model = model
        self.args = args


class PriorContrastriveEstimator(Utility):
    def __call__(
        self,
        designs=None,
        nodes=None,
        values=None,
        onehot=False,
        samples=None,
        precise=False,
        lower=True,
    ):
        n_dags = len(self.model.dags)

        if designs is not None:
            nodes = designs[..., 0]
            values = designs[..., 1]

        # DAGs x Designs x B x Samples x Nodes - y[t][m]
        if samples is None:
            samples = self.model.batch_interventional_samples(
                nodes,
                values,
                self.args.num_samples,
                deterministic=self.args.deterministic,
                onehot=onehot,
                precise=precise,
            )

            # (Pdb) samples.shape
            # (15, 1, 5, 100, 5)

        # outer x iner x batch x designs
        # B x Designs x Outer x Iner x Samples
        logprobs = self.model.batch_likelihood(
            nodes, samples, onehot=onehot, weights=False
        )

        # self.args.weighted_posterior
        # import pdb; pdb.set_trace()
        # (441, 2, 19, 19, 100)
        # self.model.log_weights[None, None, :, None]
        # batch x designs x outer x inner x samples
        num = jnp.diagonal(logprobs, axis1=2, axis2=3).sum(1).swapaxes(1, 2)

        _weighted_logprobs = (
            logprobs + self.model.log_weights[None, None, None, :, None]
        )

        if lower:
            if self.args.weighted_posterior:
                # denom = logsumexp(_weighted_logprobs.sum(1), 2)
                denom = logsumexp(
                    (
                        logprobs.sum(1)
                        + self.model.normalized_log_weights[None, None, :, None]
                    ),
                    2,
                )
                # denom = logmeanexp(logprobs.sum(1), 2)
            else:
                denom = logmeanexp(logprobs.sum(1), 2)
        else:
            B, T, D, _, S = logprobs.shape
            if self.args.weighted_posterior:
                logprobs_without_diag = _weighted_logprobs[
                    :, :, ~jnp.eye(_weighted_logprobs.shape[2], dtype=bool)
                ].reshape(B, T, D, D - 1, S)
                denom = logsumexp(logprobs_without_diag.sum(1), 2)
            else:
                logprobs_without_diag = logprobs[
                    :, :, ~jnp.eye(logprobs.shape[2], dtype=bool)
                ].reshape(B, T, D, D - 1, S)
                denom = logsumexp(logprobs_without_diag.sum(1), 2) - jnp.log(
                    logprobs_without_diag.shape[2] - 1
                )

        # import pdb; pdb.set_trace()

        if self.args.weighted_posterior:
            info_gain = (
                (
                    jnp.exp(self.model.normalized_log_weights[None, :, None])
                    * (num - denom)
                )
                .sum(1)
                .mean(1)
            )
        else:
            info_gain = (num - denom).mean((1, 2))

        return info_gain


class NestedMonteCarloEstimator(PriorContrastriveEstimator):
    def __call__(
        self,
        designs=None,
        nodes=None,
        values=None,
        onehot=False,
        samples=None,
        precise=False,
    ):
        return super().__call__(
            designs=designs,
            nodes=nodes,
            values=values,
            onehot=onehot,
            samples=samples,
            precise=precise,
            lower=False,
        )


class CovEigEstimator(Utility):
    def __call__(
        self,
        designs=None,
        nodes=None,
        values=None,
        onehot=False,
        samples=None,
        precise=False,
    ):
        n_dags = len(self.model.dags)

        if designs is not None:
            nodes = designs[..., 0]
            values = designs[..., 1]

        # DAGs x Designs x B x Samples x Nodes - y[t][m]
        if samples is None:
            samples = self.model.batch_interventional_samples(
                nodes,
                values,
                self.args.num_samples,
                deterministic=False,
                onehot=onehot,
                precise=precise,
            )

        L, T, B, S, D = samples.shape
        cov = vmap(jnp.cov)(samples.transpose([2, 1, 0, 3, 4]).reshape(B, T, -1))
        variance = max(self.args.noise_sigma)

        K = cov / variance + jnp.eye(cov.shape[1])
        return 0.5 * jnp.linalg.slogdet(K)[1]


SoftPCE_GD = partial(
    SoftTopK,
    optimizer=MultiNodeGradientBasedOptimizer,
    utility=PriorContrastriveEstimator,
)
RandOptPCE_GD = partial(
    RandomNodeOptimizeValue,
    optimizer=GradientBasedOptimizer,
    utility=PriorContrastriveEstimator,
)
SoftPCE_BO = partial(
    SoftTopK, optimizer=MultiNodeBayesianOptimizer, utility=PriorContrastriveEstimator
)
RandOptPCE_BO = partial(
    RandomNodeOptimizeValue,
    optimizer=BayesianOptimizer,
    utility=PriorContrastriveEstimator,
)
PolicyOptPCE = partial(
    PolicyOptimization,
    optimizer=GradientBasedPolicyOptimizer,
    utility=PriorContrastriveEstimator,
)
GridOptPCE = partial(
    PolicyOptimization, optimizer=GridOptimizer, utility=PriorContrastriveEstimator
)
PolicyOptNMC = partial(
    PolicyOptimization,
    optimizer=GradientBasedPolicyOptimizer,
    utility=NestedMonteCarloEstimator,
)
PolicyOptNMCFixedValue = partial(
    PolicyOptimizationWithFixedValue,
    optimizer=GradientBasedPolicyOptimizer,
    utility=NestedMonteCarloEstimator,
)
PolicyOptCovEig = partial(
    PolicyOptimization, optimizer=GradientBasedPolicyOptimizer, utility=CovEigEstimator
)
