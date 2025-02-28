# QMP: Q-switch Mixture of Policies for Multi-Task Behavior Sharing
## Accepted to ICLR 2025

[[Project Website]](https://qmp-mtrl.github.io/) [[Paper]](https://arxiv.org/abs/2302.00671) [[OpenReview]](https://openreview.net/forum?id=aUZEeb2yvK)

[Grace Zhang](https://gracehzhang.github.io/)\*<sup>1</sup>, [Ayush Jain](https://ayushj240.github.io/)\*<sup>1</sup>, [Injune Hwang]()<sup>2</sup>, [Shao-Hua Sun](https://shaohua0116.github.io/)<sup>3</sup>, [Joseph J. Lim](https://clvrai.com/web_lim/)<sup>2</sup>

<sup>1</sup>University of Southern California <sup>2</sup>KAIST <sup>3</sup>National Taiwan University

This is the official PyTorch implementation of ICLR 2025 paper **"QMP: Q-switch Mixture of Policies for Multi-Task Behavior Sharing"**. 

Abstract: QMP is a multi-task reinforcement learning approach that shares behaviors between tasks using a **mixture of policies** for off-policy data collection. We show that using the Q-function as a switch for this mixture is guaranteed to improve sample efficiency. The **Q-switch** selects which policy among the mixture that maximizes the current task's Q-value for the current state. This works because other policies might have already learned overlapping behaviors that the current task's policy has not learned. QMP's behavior sharing shows **complementary** gains over common approaches like parameter sharing and data sharing.

<p align="center">
    <img src="model.png" width="800px">
</p>

## Directories
* `run.py` take arguments and initializes experiments
* `garage_experiments.py` defines experiments and starts training
* `learning/`: contains all learning code, baseline implementations, and our method
* `environments/`: registers environments

## Dependencies
* Ubuntu 18.04 or above
* Python 3.8
* Mujoco 2.1 [https://github.com/deepmind/mujoco/releases]
## Installation

To install python dependencies.
   ```bash
   pip install -r requirements.txt
   ```

## Implementing QMP
Our implementation of QMP is based on top of the garage RL codebase [https://github.com/rlworkgroup/garage](https://github.com/rlworkgroup/garage).  If you would
like to re-implement QMP in your own codebase, it is fairly simple, as we only replace the data collection policy for each 
task, denoted $\pi_i$, with our mixture policy $\pi_i^{mix}$.  We highlight the specific changes we made in this codebase 
below, to aid in re-implementing QMP in a different codebase.

* We first initialize all the task policies and Q-function networks in the `setup` function in `experiment_utils.py`.
We then initialize the mixture policy with all the task policies and Q-functions.
* We define the mixture policy in `learning/policies/qswitch_mixture_policies_wrapper.py`.  Critically, the `get_action` function,
given an input observation and task, samples all policies for candidate actions, uses the task Q-function to evaluate the policies,
and outputs the best action.
* We pass the mixture policy to the sampler to gather data and the individual policies and Q-functions to the RL algorithm to train.


## Example Commands

To run our method in combination with other MTRL methods, follow the example commands below.  Method X and Method X + QMP are always run with the same hyperparameters.  
For data sharing, we tune `unsupervised_quantile` per task, and for parameter sharing, we increase the network size and tune the learning rates, as reported in our paper.
Simply, replace the environment name `--env=JacoReachMT5-v1`, `--env=MazeLarge-10-v0`, `--env=Walker2dMT4-v0`, `--env=MetaWorldCDS-v1`,
`--env=MetaWorldMT10-v2`,`--env=KitchenMTEasy-v0 `, or `--env=MetaWorldMT50-v2`, and update the data and parameter sharing 
hyperparameters (`unsupervised_quantile`, `lr`, `hidden_sizes`) according to the paper.

### Multistage Reacher
* Separated + QMP (Our Method)
  ```bash
  python run.py qmp_dnc --env=JacoReachMT5-v1
  ```
* Separated
  ```bash
  python run.py dnc_sac --env=JacoReachMT5-v1
  ```
* Parameters + QMP (Our Method)
  ```bash
  python run.py qmp_sac --env=JacoReachMT5-v1 --policy_architecture multihead --Q_architecture multihead --lr 0.001 --hidden_sizes 512 512 
  ```
* Parameters
  ```bash
  python run.py mtsac --env=JacoReachMT5-v1 --policy_architecture multihead --Q_architecture multihead --lr 0.001 --hidden_sizes 512 512 
  ```
* Data + QMP (Our Method)
  ```bash
  python run.py qmp_uds_dnc --env=JacoReachMT5-v1 --sharing_quantile 0
  ```
* Data
  ```bash
  python run.py uds_dnc --env=JacoReachMT5-v1 --sharing_quantile 0
  ```


## Main Results
<p align="center">
    <img src="main_results.png" width="800px">
</p>

## Citation
Please consider citing our work if you find it useful. Reach out to us for any questions!
```
@inproceedings{
zhang2025qmp,
title={{QMP}: Q-switch Mixture of Policies for Multi-Task Behavior Sharing},
author={Grace Zhang and Ayush Jain and Injune Hwang and Shao-Hua Sun and Joseph J Lim},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=aUZEeb2yvK}
}
```
