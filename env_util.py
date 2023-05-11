import numpy as np


def step(env, state, action):
    flat_probs = np.ndarray.flatten(env.transitions[state, action])
    next_state = np.random.choice(env.nb_states, 1, p=flat_probs)
    reward = env.rewards[state, action]
    done = False
    if next_state in env.terminal_states:
        done = True
    return next_state, reward, done


def fast_step(env, state, action):
    flat_probs = np.ndarray.flatten(env.transitions[state, action])
    cumsum = np.cumsum(flat_probs)
    rdm_unif = np.random.rand(1)
    next_state = np.searchsorted(cumsum, rdm_unif)
    reward = env.rewards[state, action]
    done = False
    if next_state in env.terminal_states:
        done = True
    return next_state, reward, done



def generate_batch(env, nb_trajectories, pi, max_steps=50):
    trajectories = []
    for _ in range(nb_trajectories):
        nb_steps = 0
        trajectorY = []
        state = env.init_state
        done = False
        while nb_steps < max_steps and not done:
            action = np.random.multinomial(1, np.ndarray.flatten(pi[state])).argmax()
            # action_choice = np.random.choice(pi.shape[1], p=pi[state])
            next_state, reward, done = fast_step(env, state, action)
            trajectorY.append([action, state, next_state, reward])
            state = next_state
            nb_steps += 1
        trajectories.append(trajectorY)
    batch_traj = [val for sublist in trajectories for val in sublist]
    return trajectories, batch_traj
