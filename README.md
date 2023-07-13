[![arXiv Badge](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=fff&style=for-the-badge)](https://arxiv.org/abs/2302.10607) 


# Differentiable Multi-Target Causal Bayesian Experimental Design (DiffCBED)

## Overview 

This branch contains the commands that would help reproduce the results in the paper. For any questions regarding this code, please contact at yashas.annadani@gmail.com or ptigas@robots.ox.ac.uk.

## Reproducing the results 

### Single-Target
#### DiffCBED
`bash run_experiments.sh --model dag_bootstrap --num_nodes 50 --batch_size 5 --num_starting_samples 60 --strategy policyoptnmc_fixed --no_sid --group_interventions --num_particles 30 --num_samples 25`

#### SoftCBED
`bash run_experiments.sh --model dag_bootstrap --num_nodes 50 --batch_size 5 --num_starting_samples 60 --strategy softbald --value_strategy gp-ucb --no_sid --group_interventions --num_particles 30 --num_samples 25`

#### Random Random
`bash run_experiments.sh --model dag_bootstrap --num_nodes 50 --batch_size 5 --num_starting_samples 60 --strategy random --value_strategy random --no_sid --group_interventions --num_particles 30`

#### Random Fixed
`bash run_experiments.sh --model dag_bootstrap --num_nodes 50 --batch_size 5 --num_starting_samples 60 --strategy random --value_strategy fixed --no_sid --group_interventions --num_particles 30`

### Constrained Multi-Target
#### DiffCBED
`bash run_experiments.sh --model dag_bootstrap --num_nodes 40 --exp_edges 0.05 --batch_size 2 --num_starting_samples 800 --strategy policyoptnmc_fixed --no_sid --group_interventions --num_particles 30 --num_samples 25 --num_targets 5 --num_batches 1 --include_gt_mec --reuse_posterior_samples`

#### SSGb
`bash run_experiments.sh --model dag_bootstrap --num_nodes 40 --exp_edges 0.05 --batch_size 2 --num_starting_samples 800 --strategy ss_finite --no_sid --group_interventions --save_path results --num_particles 30 --num_samples 25 --num_targets 5 --num_batches 1 --include_gt_mec --reuse_posterior_samples --intervention_value 5`

#### Random Fixed
`bash run_experiments.sh --model dag_bootstrap --num_nodes 40 --exp_edges 0.05 --batch_size 2 --num_starting_samples 800 --strategy random --value_strategy fixed --no_sid --group_interventions --save_path results --num_particles 30 --num_samples 25 --num_targets 5 --num_batches 1 --include_gt_mec --reuse_posterior_samples --intervention_value 5`

### Unconstrained Multi-Target
#### DiffCBED
`bash run_experiments.sh --model dag_bootstrap --num_nodes 20 --batch_size 2 --num_starting_samples 60 --strategy policyoptnmc_fixed --no_sid --group_interventions --save_path results --num_particles 30 --num_samples 25 --exp_edges .5 --intervention_value 5 --num_targets -1 --old_er_logic`

#### SSGb
`bash run_experiments.sh --model dag_bootstrap --num_nodes 20 --batch_size 2 --num_starting_samples 60 --strategy ss_finite --no_sid --group_interventions --save_path results_20_s60_old_er_5_b2_fixed_ss --num_particles 30 --num_samples 25 --exp_edges .5 --intervention_value 5 --num_targets 20 --old_er_logic`

#### Random Fixed
`bash run_experiments.sh --model dag_bootstrap --num_nodes 20 --batch_size 2 --num_starting_samples 60 --strategy random --value_strategy fixed --no_sid --group_interventions --save_path results_20_s60_old_er_5_b2 --num_particles 30 --num_samples 25 --exp_edges .5 --intervention_value 5 --num_targets -1 --old_er_logic`

## Reference
This code is official implementation of the following paper:
> Yashas Annadani, Panagiotis Tigas, Desi R. Ivanova, Andrew Jesson, Yarin Gal, Adam Foster, Stefan Bauer. **Differentiable Multi-Target Causal Bayesian Experimental Design**. In International Conference on Machine Learning (ICML), 2023. [PDF](https://arxiv.org/pdf/2302.10607.pdf)
