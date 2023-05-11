# authors: anonymized

import os
import sys
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4

import argparse
from multiprocessing import Pool

from math import ceil, floor, sqrt, log
from scipy import special as sc
import pandas as pd
import numpy as np
from yaml import dump

import spibb_utils
import mazeDiscrete
import spibb
import modelTransitions
import garnets
from guarantee_functions import compute_zeta_spibb, compute_Nwedge_2s, compute_Nwedge_beta


DEFAULT_SEEDS = [1]
DEFAULT_DATASET_SIZES = [1, 2, 5, 8, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
DEFAULT_NWEDGE = 200

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expname",
        help="name of the experiment"
    )
    parser.add_argument(
        "seeds",
        type=int,
        nargs="+",
        help="list of seeds"
    )
    parser.add_argument(
        "-n", "--Nwedges",
        default=DEFAULT_NWEDGE,
        type=int,
        nargs="+",
        help="Nwedge parameters for standard SPIBB"
    )
    parser.add_argument(
        "-c", "--cpus",
        default=1,
        type=int,
        help="number of cpus to par"
    )
    parser.add_argument(
        "--nb_trajectories_list",
        default=DEFAULT_DATASET_SIZES,
        type=int,
        nargs="+",
        help="number of worker processes"
    )
    parser.add_argument(
        "--output_path",
        default="results/",
        type=str,
        help="output directory (default=results)"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="runs on verbose mode"
    )

    return parser.parse_args()


def safe_save(filename, df):
    df.to_csv(filename + '.csv')
    spibb_utils.prt(f'{df.size} lines saved to {filename}.csv')


def main(seed, expname, Nwedges, nb_trajectories_list, output_path, verbose):
    Nwedge = Nwedges[0]

    spibb_utils.prt('Start of experiment')
    np.random.seed(seed)

    # Definition of the environment
    x_max = 5
    y_max = 5
    x_end = int(x_max - 1)
    y_end = int(y_max - 1)
    walls = [[[2.5, 0], [2.5, 3]], [[3.5, 2], [3.5, 4]]]
    walls = [[[x_max/2., 0], [x_max/2., ceil(y_max/2.)]], [[x_max/2.+1., floor(y_max/2.)], [x_max/2.+1., ceil(y_max/2.)+1.]]]
    maze = mazeDiscrete.Maze(x_max, y_max, walls, x_end, y_end, env_type=0)
    nb_states = int(x_max * y_max)
    if maze.env_type == 1:
        nb_states = nb_states * 2
    nb_actions = 4

    # Definition of the objective function:
    gamma = 0.95
    # Load the baseline policy state-action function
    npy_filename = "state_action_val_used_size_" + str(int(x_max)) + "_env_type_0.npy"

    Q_baseline = np.load(npy_filename)

    # Compute the baseline policy:
    pi_b = spibb_utils.compute_baseline(Q_baseline)


    delta = 0.1
    vmax = 0.6
    zeta = 0.1

    v = np.zeros(nb_states)

    # Pre-compute the true reward function in function of SxA:
    current_proba = maze.transition_function
    garnet = garnets.Garnets(nb_states, nb_actions, 1, self_transitions=0)
    garnet.transition_function = current_proba
    reward_current = garnet.compute_reward()
    r_reshaped = spibb_utils.get_reward_model(current_proba, reward_current)


    # Compute the baseline policy performance:
    pi_b_perf = spibb.policy_evaluation_exact(pi_b, r_reshaped, current_proba, gamma)[0][0]
    #print("baseline_perf: " + str(pi_b_perf))


    # Creates a mask that is always True for classical RL and other non policy-based SPIBB algorithms# mask_0 = ~ spibb.compute_mask(nb_states, nb_actions, 1, 1, [])
    mask_0, thres = spibb.compute_mask(nb_states, nb_actions, 1, 1, [])
    mask_0 = ~mask_0

    pi_star = spibb.spibb(gamma, nb_states, nb_actions, mask_0, mask_0, current_proba, r_reshaped, 'default')
    pi_star.fit()
    pi_star_perf = spibb.policy_evaluation_exact(pi_star.pi, r_reshaped, current_proba, gamma)[0][0]
    #print("pi_star_perf: " + str(pi_star_perf))


    results = dict()
    for Nwedge in Nwedges:
        results[Nwedge] = []
    Nwedges_2s = dict()
    Nwedges_beta = dict()

    for nb_trajectories in nb_trajectories_list:
        # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples
        trajectories, batch_traj = spibb_utils.generate_batch(nb_trajectories, garnet, pi_b)
        spibb_utils.prt("GENERATED A DATASET OF " + str(nb_trajectories) + " TRAJECTORIES")

        # Compute the maximal likelihood model for transitions and rewards.
        # NB: the true reward function can be used for ease of implementation since it is not stochastic in our environment.
        # One should compute it fro mthe samples when it is stochastic.
        model = modelTransitions.ModelTransitions(batch_traj, nb_states, nb_actions)
        reward_model = spibb_utils.get_reward_model(model.transitions, reward_current)

        # Computes the RL policy
        rl = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_0, model.transitions, reward_model, 'default')
        rl.fit()
        # Evaluates the RL policy performance
        perfrl = spibb.policy_evaluation_exact(rl.pi, r_reshaped, current_proba, gamma)[0][0]
        #print("perf RL: " + str(perfrl))

        for Nwedge in Nwedges:
            N_wedge_spibb = Nwedge

            # Computation of the binary mask for the bootstrapped state actions
            mask = spibb.compute_mask_N_wedge(nb_states, nb_actions, N_wedge_spibb, batch_traj)
            # Computation of the model mask for the bootstrapped state actions
            # masked_model = model.masked_model(mask)

            # Computes the Pi_b_SPIBB policy:
            pib_SPIBB = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Pi_b_SPIBB')
            pib_SPIBB.fit()
            # Evaluates the Pi_b_SPIBB performance:
            perf_Pi_b_SPIBB = spibb.policy_evaluation_exact(pib_SPIBB.pi, r_reshaped, current_proba, gamma)[0][0]
            #print("perf Pi_b_SPIBB: " + str(perf_Pi_b_SPIBB))



            zeta_spibb = compute_zeta_spibb(N_wedge_spibb, delta, gamma, vmax, nb_states, nb_actions, pi_b_perf, pi_star_perf)
            N_wedge_2S = compute_Nwedge_2s(zeta_spibb, delta, gamma, vmax, nb_states, nb_actions, pi_b_perf, pi_star_perf)
            N_wedge_beta = compute_Nwedge_beta(zeta_spibb, delta, gamma, vmax, nb_states, nb_actions, N_wedge_2S, pi_b_perf, pi_star_perf)

            Nwedges_2s[Nwedge] = N_wedge_2S
            Nwedges_beta[Nwedge] = N_wedge_beta

            # Computes the Pi_b_SPIBB 2S policy:
            mask_new = spibb.compute_mask_N_wedge(nb_states, nb_actions, N_wedge_2S, batch_traj)
            masked_model = model.masked_model(mask_new)

            # Computes the Pi_b_SPIBB policy:
            pib_SPIBB_2s = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_new, model.transitions, reward_model, 'Pi_b_SPIBB')
            pib_SPIBB_2s.fit()
            # Evaluates the Pi_b_SPIBB performance:
            perf_2S = spibb.policy_evaluation_exact(pib_SPIBB_2s.pi, r_reshaped, current_proba, gamma)[0][0]
            #print("perf 2S: " + str(perf_2S))



            # Computes the Pi_b_SPIBB beta policy:
            mask_new = spibb.compute_mask_N_wedge(nb_states, nb_actions, N_wedge_beta, batch_traj)
            masked_model = model.masked_model(mask_new)

            # Computes the Pi_b_SPIBB policy:
            pib_SPIBB_beta = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_new, model.transitions, reward_model,
                                       'Pi_b_SPIBB')
            pib_SPIBB_beta.fit()
            # Evaluates the Pi_b_SPIBB performance:
            perf_beta = spibb.policy_evaluation_exact(pib_SPIBB_beta.pi, r_reshaped, current_proba, gamma)[0][0]
            #print("perf beta: " + str(perf_beta))

            results[Nwedge].append([seed, nb_trajectories, "Basic RL", perfrl])
            results[Nwedge].append([seed, nb_trajectories, "Pi_b_SPIBB", perf_Pi_b_SPIBB])
            results[Nwedge].append([seed, nb_trajectories, "2S", perf_2S])
            results[Nwedge].append([seed, nb_trajectories, "Beta", perf_beta])

    for Nwedge in Nwedges:
        if seed == 1:
            parameters = dict(
                zip(['seed', 'gamma', 'nb_states', 'nb_actions', 'nb_next_state_transition', 'baseline_perf',
                     'pi_star_perf',
                     'N_wedge_spibb', 'N_wedge_2S', 'N_wedge_beta'],
                    [seed, gamma, nb_states, nb_actions, 4, float(pi_b_perf), float(pi_star_perf), Nwedge,
                     Nwedges_2s[Nwedge], Nwedges_beta[Nwedge]]))
            with open(os.path.join('results/' + expname + '/Nwedge_' + str(Nwedge) + '/config.yaml'), 'w') as f:
                dump(data=parameters, stream=f)

        # Place to save the results
        filename = f'{output_path}/{expname}/Nwedge_{Nwedge}/results_{seed}'

        df = pd.DataFrame(results[Nwedge], columns=['seed', 'nb_trajectories', 'algorithm', 'performance'])

        # Save it to a csv file:
        safe_save(filename, df)
    return True


if __name__ == '__main__':
    args = vars(parse_args())
    seeds = args.pop("seeds")
    cpus = args.pop("cpus")

    Nwedges = args['Nwedges']
    expname = args['expname']
    output_path = args['output_path']

    for Nwedge in Nwedges:
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        if not os.path.isdir(f'{output_path}/{expname}'):
            os.mkdir(f'{output_path}/{expname}')
        if not os.path.isdir(f'{output_path}/{expname}/Nwedge_{Nwedge}'):
            os.mkdir(f'{output_path}/{expname}/Nwedge_{Nwedge}')

    def f(s):
        main(s, **args)

    if cpus > 1:
        with Pool(cpus) as p:
            p.map(f, seeds)
    else:
        for s in seeds:
            f(s)

    print("DONE")
