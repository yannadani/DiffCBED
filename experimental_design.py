import argparse
import json
import os
import random
from csv import reader

import numpy as np
import pandas as pd

from envs import Chain, ErdosRenyi, OnlyDAGDream4Environment, ScaleFree
from envs.samplers import Constant
from models import DagBootstrap
from replay_buffer import ReplayBuffer
from strategies import (
    ABCDStrategy,
    BALDStrategy,
    BatchBALDStrategy,
    FScoreBatchStrategy,
    GreedyABCDStrategy,
    GridOptPCE,
    PCEBatchStrategy,
    PolicyOptCovEig,
    PolicyOptNMC,
    PolicyOptNMCFixedValue,
    PolicyOptPCE,
    RandomAcquisitionStrategy,
    RandOptPCE_BO,
    RandOptPCE_GD,
    ReplayStrategy,
    SoftBALDStrategy,
    SoftFScoreStrategy,
    SoftPCE_BO,
    SoftPCE_GD,
    SSFinite,
)
from utils.logger import Logger

wandb = None
if "WANDB_API_KEY" in os.environ:
    import wandb

import warnings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Differentiable Multi-Target Causal Bayesian Experimental Design"
    )
    parser.add_argument(
        "--save_path", type=str, default="results/", help="Path to save result files"
    )
    parser.add_argument("--id", type=str, default=None, help="ID for the run")
    parser.add_argument(
        "--data_seed",
        type=int,
        default=20,
        help="random seed for generating data (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--num_nodes", type=int, default=5, help="Number of nodes in the causal model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dag_bootstrap",
        help="Posterior model to use {dag_bootstrap}",
    )
    parser.add_argument("--env", type=str, default="erdos", help="SCM to use")
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        help="Acqusition strategy to use {abcd, random}",
    )
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument(
        "--sparsity_factor",
        type=float,
        default=0.0,
        help="Hyperparameter for sparsity regulariser",
    )
    parser.add_argument(
        "--exp_edges",
        type=float,
        default=0.1,
        help="probability of expected edges in random graphs",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Total number of samples in the synthetic data",
    )
    parser.add_argument(
        "--num_targets",
        type=int,
        default=1,
        help="Total number of targets.",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=30,
        help="Total number of posterior samples",
    )
    parser.add_argument(
        "--num_starting_samples",
        type=int,
        default=100,
        help="Total number of samples in the synthetic data to start with",
    )
    parser.add_argument(
        "--temperature_type",
        type=str,
        default="anneal",
        help="Whether to anneal the relaxed distribution temperature or keep it fixed",
    )
    parser.add_argument(
        "--exploration_steps",
        type=int,
        default=1,
        help="Total number of exploration steps in gp-ucb",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="isotropic-gaussian",
        help="Type of noise of causal model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature of soft bald/ reparameterized sampling",
    )
    parser.add_argument(
        "--noise_sigma", type=float, default=0.1, help="Std of Noise Variables"
    )
    parser.add_argument(
        "--scm_bias",
        type=float,
        default=0.0,
        help="Bias term of the additive gaussian noise.",
    )
    parser.add_argument(
        "--theta_mu", type=float, default=2.0, help="Mean of Parameter Variables"
    )
    parser.add_argument(
        "--theta_sigma", type=float, default=1.0, help="Std of Parameter Variables"
    )
    parser.add_argument(
        "--gibbs_temp", type=float, default=1000.0, help="Temperature of Gibbs factor"
    )

    # TODO: improve names
    parser.add_argument(
        "--num_intervention_values",
        type=int,
        default=5,
        help="Number of interventional values to consider.",
    )
    parser.add_argument(
        "--intervention_values",
        type=float,
        nargs="+",
        help="Interventioanl values to set in `grid` value_strategy, else ignored.",
    )
    parser.add_argument(
        "--intervention_value",
        type=float,
        default=0.0,
        help="Interventional value to set in `fixed` value_strategy, else ingored.",
    )

    parser.add_argument("--group_interventions", action="store_true")
    parser.add_argument("--plot_graphs", action="store_true")
    parser.add_argument("--save_models", action="store_true", default=False)
    parser.set_defaults(group_interventions=False)
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--old_er_logic", action="store_true")
    parser.set_defaults(nonlinear=False)
    parser.add_argument("--weighted_posterior", action="store_true")
    parser.set_defaults(weighted_posterior=False)
    parser.add_argument("--reuse_posterior_samples", action="store_true")
    parser.set_defaults(reuse_posterior_samples=False)
    parser.add_argument("--include_gt_mec", action="store_true")
    parser.set_defaults(include_gt_mec=False)
    parser.add_argument(
        "--value_strategy",
        type=str,
        default="fixed",
        help="Possible strategies: gp-ucb, grid, fixed, sample-dist",
    )
    parser.add_argument(
        "--dream4_path",
        type=str,
        default="envs/dream4/configurations/",
        help="Path of DREAM4 files.",
    )
    parser.add_argument(
        "--dream4_name",
        type=str,
        default="insilico_size10_1",
        help="Name of DREAM4 experiment to load.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save_graphs", action="store_true")

    # optimizers
    parser.add_argument(
        "--opt_lr",
        type=float,
        default=0.5,
        help="Learning rate of the gradient based optimizer.",
    )
    parser.add_argument(
        "--opt_epochs",
        type=int,
        default=100,
        help="Epochs for the gradient based optimizers",
    )

    parser.add_argument(
        "--node_range", default="-10:10", help="Node value ranges (constraints)"
    )

    parser.set_defaults(nonlinear=False)

    args = parser.parse_args()
    args.node_range = [float(item) for item in args.node_range.split(":")]

    if args.env == "sf":
        args.dibs_graph_prior = "sf"

    # if args.nonlinear == False:
    #     args.group_interventions = True

    if args.include_gt_mec:
        print("Setting weighted_posterior to True because include_gt_mec is set")
        args.weighted_posterior = True

    if args.reuse_posterior_samples:
        print(
            "Setting weighted_posterior to True because reuse_posterior_samples is set"
        )
        args.weighted_posterior = True

    return args


STRATEGIES = {
    "greedyabcd": GreedyABCDStrategy,
    "abcd": ABCDStrategy,
    "softbald": SoftBALDStrategy,
    "batchbald": BatchBALDStrategy,
    "bald": BALDStrategy,
    "random": RandomAcquisitionStrategy,
    "replay": ReplayStrategy,
    "f-score": FScoreBatchStrategy,
    "softf-score": SoftFScoreStrategy,
    "pce": PCEBatchStrategy,
    "softpce_bo": SoftPCE_BO,
    "randoptpce_bo": RandOptPCE_BO,
    "softpce_gd": SoftPCE_GD,
    "randoptpce_gd": RandOptPCE_GD,
    "policyoptpce": PolicyOptPCE,
    "ss_finite": SSFinite,
    "policyoptnmc": PolicyOptNMC,
    "policyoptnmc_fixed_value": PolicyOptNMCFixedValue,
    "gridpce": GridOptPCE,
    "policyoptcoveig": PolicyOptCovEig,
}

MODELS = {
    "dag_bootstrap": DagBootstrap,
}

ENVS = {
    "erdos": ErdosRenyi,
    "chain": Chain,
    "sf": ScaleFree,
    "semidream4": OnlyDAGDream4Environment,
}


def causal_experimental_design_loop(args):
    # prepare save path
    args.save_path = os.path.join(
        args.save_path,
        "_".join(
            map(
                str,
                [
                    args.env,
                    args.data_seed,
                    args.seed,
                    args.num_nodes,
                    args.num_starting_samples,
                    args.model,
                    args.strategy,
                    args.value_strategy,
                    args.exp_edges,
                    args.noise_type,
                    args.noise_sigma,
                    args.temperature,
                    args.temperature_type,
                    args.intervention_value,
                    "nonlinear" if args.nonlinear else "linear",
                    args.dream4_name if args.env == "dream4" else "",
                    args.id,
                ],
            )
        ),
    )

    if wandb is not None:
        wandb.init(project="CED", name=args.id)
        wandb.config.update(args, allow_val_change=True)

    os.makedirs(args.save_path, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.save_path, "config.json"), "w"))
    logger = Logger(args.save_path, resume=args.resume, wandb=wandb)

    # set the seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.env == "semidream4":
        env = ENVS[args.env](
            args=args,
            noise_type=args.noise_type,
            noise_sigma=args.noise_sigma,
            num_samples=args.num_samples,
            node_range=args.node_range,
            mu_prior=args.theta_mu,
            sigma_prior=args.theta_sigma,
            seed=args.data_seed,
            path=args.dream4_path,
            name=args.dream4_name,
            nonlinear=args.nonlinear,
            binary_nodes=True,
            logger=logger,
        )
        args.num_nodes = env.num_nodes
        args.noise_sigma = [args.noise_sigma] * args.num_nodes
    else:
        env = ENVS[args.env](
            args=args,
            num_nodes=args.num_nodes,
            exp_edges=args.exp_edges,
            noise_type=args.noise_type,
            noise_sigma=args.noise_sigma,
            num_samples=args.num_samples,
            node_range=args.node_range,
            mu_prior=args.theta_mu,
            sigma_prior=args.theta_sigma,
            seed=args.data_seed,
            nonlinear=args.nonlinear,
            binary_nodes=True,
            logger=logger,
        )
        args.noise_sigma = env._noise_std

    model = MODELS[args.model](env, args)

    p_edges = env.adjacency_matrix.mean()
    # expected_edges = p_edges*(env.adjacency_matrix.shape[0]-1)
    expected_edges = env.adjacency_matrix.sum() / env.adjacency_matrix.shape[0]
    print(
        f"p of edge: {p_edges} num edges:{env.adjacency_matrix.sum()} expected edges: {expected_edges}"
    )
    # import pdb; pdb.set_trace()

    env.plot_graph(os.path.join(args.save_path, "graph.png"))

    buffer = ReplayBuffer(binary=True)
    # sample num_starting_samples initially - not num_samples
    buffer.update(env.sample(args.num_starting_samples))

    # Sample held out observational samples to evaluate log likelihood of the model
    held_out_data = env.held_out_data

    # if DAG_BOOTSTRAP:
    samples = buffer.data().samples
    args.sample_mean = samples.mean(0)
    args.sample_std = samples.std(0, ddof=1)

    model.covariance_matrix = np.cov(samples.T)

    strategy = STRATEGIES[args.strategy](model, env, args)

    # evaluate
    start_batch = 0
    no_file = 0
    if args.resume:
        try:
            _metrics = pd.read_json(open(logger.metrics_path), lines=True)
            start_batch = int(
                _metrics["interventional_samples"].max() / args.batch_size
            )
            if start_batch > 0:
                with open(logger.interventions_path, "r") as f:
                    csv_reader = reader(f)
                    for row in csv_reader:
                        buffer.update(
                            env.intervene(
                                int(row[0]),
                                1,
                                int(row[1]),
                                Constant(float(row[2])),
                                _log=False,
                            )
                        )
            model.load(args.save_path)
        except:
            no_file = 1
    if not args.resume or no_file or start_batch == 0:
        # evaluate

        model.update(buffer.data())

        logger.log_metrics(
            {
                "eshd": env.eshd(model, 1000, double_for_anticausal=False).item(),
                "i_mmd": env.i_mmd(model).item(),
                "f1_score": env.f1_score(model).item(),
            }
        )
        if args.save_graphs:
            logger.log_graphs(iteration=-1, model=model)
        if args.save_models:
            os.makedirs(args.save_path + f"/models/iteration_{0}", exist_ok=True)
            model.save(args.save_path + f"/models/iteration_{0}")

    warnings.warn(
        "Assuming value sampler corresponds to `Constant` distribution. Code can break/could lead to wrong results if using any other sampler"
    )
    for i in range(start_batch, args.num_batches):
        print(f"====== Experiment {i+1} =======")

        strategy = STRATEGIES[args.strategy](model, env, args)

        # example of information based strategy
        valid_interventions = list(range(args.num_nodes))
        interventions, _ = strategy.acquire(valid_interventions, i)

        for k in range(args.batch_size):
            buffer.update(
                env.intervene(
                    i, 1, interventions["nodes"][k], interventions["values"][k]
                )
            )

        model.update(buffer.data())

        logger.log_metrics(
            {
                "eshd": env.eshd(model, 1000, double_for_anticausal=False).item(),
                "i_mmd": env.i_mmd(model).item(),
                "f1_score": env.f1_score(model).item(),
            }
        )

        if args.save_models:
            os.makedirs(args.save_path + f"/models/iteration_{i+1}", exist_ok=True)
            model.save(args.save_path + f"/models/iteration_{i+1}")

        if args.save_graphs:
            logger.log_graphs(iteration=i, model=model)
        if wandb is not None:
            wandb.save(args.save_path + "/interventions.csv")


if __name__ == "__main__":
    args = parse_args()
    causal_experimental_design_loop(args)
