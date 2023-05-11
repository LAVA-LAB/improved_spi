# authors: anonymized
import os
import sys
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4

import argparse
#from multiprocessing import Pool
import multiprocess as mp

from math import ceil, floor, sqrt, log
from scipy import special as sc
import pandas as pd
import numpy as np
from yaml import dump

import spibb_utils
import env_util
import mazeDiscrete
import spibb
import modelTransitions
import garnets
from guarantee_functions import compute_zeta_spibb, compute_Nwedge_2s, compute_Nwedge_beta
import qlearning
import wetChickenEnvironment
import resourceGathering


DEFAULT_SEEDS = [1]
DEFAULT_DATASET_SIZES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
DEFAULT_NWEDGE = 200
DEFAULT_ENV = "chicken"

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
        "-e", "--environment",
        default=DEFAULT_ENV,
        type=str,
        help="environment to run; choose from [chain, river]"
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
   # parser.add_argument(
   #     "--nb_trajectories_list",
   #     default=DEFAULT_DATASET_SIZES,
   #     type=int,
   #     nargs="+",
   #     help="number of worker processes"
   # )
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


def set_environment(environment):
    if environment == "grid":
        print("not implemented here - use gridworld_main.py instead")
        exit()
    if environment == "chicken":
        nb_trajectories_list = [1, 2, 5, 8, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]#, 50000, 100000]
        env = wetChickenEnvironment.Chicken()
        if not os.path.isfile("chicken_policy.npy"):
            learner = qlearning.Qlearning(env)
            learner.learn(50, 100)
            policy = learner.softmax_policy()
            learner.save_policy(policy, "chicken_policy.npy")

        nb_states = env.nb_states
        nb_actions = env.nb_actions
        transitions = env.transitions
        rewards = env.rewards

        # Definition of the objective function:
        gamma = 0.95
        delta = 0.1
        vmax = ceil(rewards.max() / (1 - gamma))
        # Load the baseline policy state-action function

        Q_baseline = np.load("chicken_policy.npy")

        return env, nb_states, nb_actions, transitions, rewards, gamma, delta, vmax, Q_baseline, nb_trajectories_list, 10
    if environment == "resource":
        nb_trajectories_list = [1, 2, 5, 8, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]#, 50000]#, 50000, 100000]
        #nb_trajectories_list = [1, 2, 5, 8]
        env = resourceGathering.ResourceGathering()

        nb_states = env.nb_states
        nb_actions = env.nb_actions
        transitions = env.transitions
        rewards = env.rewards

        # Definition of the objective function:
        gamma = 0.995
        delta = 0.1
        vmax = ceil(rewards.max() / (1 - gamma))


        if not os.path.isfile("resource_policy.npy"):
            current_proba = transitions
            r_reshaped = rewards
            mask_0, thres = spibb.compute_mask(nb_states, nb_actions, 1, 1, [])
            mask_0 = ~mask_0
            pi_star = spibb.spibb(gamma, nb_states, nb_actions, mask_0, mask_0, current_proba, r_reshaped, 'default')
            pi_star.fit()
            pi_star_perf = spibb.policy_evaluation_exact(pi_star.pi, r_reshaped, current_proba, gamma)[0][0]

            # modify pi_star
            policy = pi_star.pi
            alpha = 0.00001
            for s in range(env.nb_states):
                a_opt = np.argmax(policy[s])
                policy[s, a_opt] = 1.0 - alpha*(env.nb_actions-1)
                for a in range(env.nb_actions):
                    if not a == a_opt:
                        policy[s, a] = alpha
            np.save("resource_policy.npy", policy)


        # Load the baseline policy state-action function
        Q_baseline = np.load("resource_policy.npy")

        return env, nb_states, nb_actions, transitions, rewards, gamma, delta, vmax, Q_baseline, nb_trajectories_list, 60
    else:
        print("ERROR UNKNOWN ENVIRONMENT SPECIFIED")
        exit(-1)




def main(seed, expname, environment, Nwedges, output_path, verbose):
    Nwedge = Nwedges[0]

    spibb_utils.prt('Start of experiment')
    np.random.seed(seed)

    env, nb_states, nb_actions, transitions, rewards, gamma, delta, vmax, Q_baseline, nb_trajectories_list, max_steps = set_environment(environment)
    # Definition of the environment


    # Compute the baseline policy:
    pi_b = spibb_utils.compute_baseline(Q_baseline)

    # The batch sizes:
    #const = 8 * vmax**2 / (zeta**2 * (1-gamma)**2)
    # const = 1

    #N_wedge_spibb = floor(const * (np.log((2 * nb_states * nb_actions * 2 ** nb_states) / delta)))
    #N_wedge_2S = floor(const * (np.log((8 * nb_states ** 2 * nb_actions ** 2) / delta)))

    v = np.zeros(nb_states)

    # Pre-compute the true reward function in function of SxA:
    current_proba = transitions
    reward_current = rewards
    r_reshaped = rewards


    # Compute the baseline policy performance:
    pi_b_perf = spibb.policy_evaluation_exact(pi_b, r_reshaped, current_proba, gamma)[0][0]
    #print("pi_b_perf", pi_b_perf)
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

    #print(pi_b_perf)
    #parameters = dict(zip(['seed', 'gamma', 'nb_states', 'nb_actions', 'nb_next_state_transition', 'baseline_perf', 'pi_star_perf', 'N_wedge_spibb', 'N_wedge_2S'],
    #[seed, gamma, nb_states, nb_actions, 4, float(pi_b_perf), float(pi_star_perf), N_wedge_spibb, -1]))
    #print(parameters)

    #with open(os.path.join('results/' + expname + '/config.yaml'), 'w') as f:
    #    dump(data=parameters, stream=f)

    for nb_trajectories in nb_trajectories_list:
        # Generate trajectories, both stored as trajectories and (s,a,s',r) transition samples

        trajectories, batch_traj = env_util.generate_batch(env, nb_trajectories, pi_b, max_steps=max_steps)
        spibb_utils.prt("GENERATED A DATASET OF " + str(nb_trajectories) + " TRAJECTORIES")

        # Compute the maximal likelihood model for transitions and rewards.
        # NB: the true reward function can be used for ease of implementation since it is not stochastic in our environment.
        # One should compute it from the samples when it is stochastic.
        model = modelTransitions.ModelTransitions(batch_traj, nb_states, nb_actions)
        reward_model = rewards

        # Computes the RL policy
        rl = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask_0, model.transitions, reward_model, 'default')
        rl.fit()
        # Evaluates the RL policy performance
        perfrl = spibb.policy_evaluation_exact(rl.pi, r_reshaped, current_proba, gamma)[0][0]
        #print("perf RL: " + str(perfrl))


        #
        # mask = spibb.compute_mask_N_wedge(nb_states, nb_actions, N_wedge_spibb, batch_traj)
        # # masked_model = model.masked_model(mask)
        #
        # ## Policy-based SPIBB ##
        #
        # # Computes the Pi_b_SPIBB policy:
        # pib_SPIBB = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Pi_b_SPIBB')
        # pib_SPIBB.fit()
        # # Evaluates the Pi_b_SPIBB performance:
        # perf_Pi_b_SPIBB = spibb.policy_evaluation_exact(pib_SPIBB.pi, r_reshaped, current_proba, gamma)[0][0]
        # print("perf Pi_b_SPIBB: " + str(perf_Pi_b_SPIBB))
        #

        for Nwedge in Nwedges:


            # Computation of the binary mask for the bootstrapped state actions
            mask = spibb.compute_mask_N_wedge(nb_states, nb_actions, Nwedge, batch_traj)
            # Computation of the model mask for the bootstrapped state actions
            # masked_model = model.masked_model(mask)

            # Computes the Pi_b_SPIBB policy:
            pib_SPIBB = spibb.spibb(gamma, nb_states, nb_actions, pi_b, mask, model.transitions, reward_model, 'Pi_b_SPIBB')
            pib_SPIBB.fit()
            # Evaluates the Pi_b_SPIBB performance:
            perf_Pi_b_SPIBB = spibb.policy_evaluation_exact(pib_SPIBB.pi, r_reshaped, current_proba, gamma)[0][0]
            #print("perf Pi_b_SPIBB: " + str(perf_Pi_b_SPIBB))



            zeta_spibb = compute_zeta_spibb(Nwedge, delta, gamma, vmax, nb_states, nb_actions, pi_b_perf, pi_star_perf)
            N_wedge_2S = compute_Nwedge_2s(zeta_spibb, delta, gamma, vmax, nb_states, nb_actions, pi_b_perf, pi_star_perf)
            N_wedge_beta = compute_Nwedge_beta(zeta_spibb, delta, gamma, vmax, nb_states, nb_actions, N_wedge_2S, pi_b_perf, pi_star_perf)

            Nwedges_2s[Nwedge] = N_wedge_2S
            Nwedges_beta[Nwedge] = N_wedge_beta

            #print("Nwedge_2S: ", N_wedge_2S)
            #print("Nwedge_beta: ", N_wedge_beta)

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
                zip(['seed', 'gamma', 'nb_states', 'nb_actions', 'nb_next_state_transition', 'baseline_perf', 'pi_star_perf',
                     'N_wedge_spibb', 'N_wedge_2S', 'N_wedge_beta'],
                    [seed, gamma, nb_states, nb_actions, 4, float(pi_b_perf), float(pi_star_perf), Nwedge, Nwedges_2s[Nwedge], Nwedges_beta[Nwedge]]))
            with open(os.path.join('results/' + expname + '/Nwedge_'+ str(Nwedge) +'/config.yaml'), 'w') as f:
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
        return main(s, **args)

    if cpus > 1:
        with mp.Pool(processes=cpus) as pool:
            result = pool.map(f, seeds)
    else:
        for s in seeds:
            f(s)

    print("DONE")
