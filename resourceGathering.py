import numpy as np
import spibb_utils

class ResourceGathering():

    def __init__(self):

        self.transitions = self.set_transitions()
        self.rewards = self.set_rewards()
        self.init_state = 0
        self.terminal_states = []
        self.nb_states = self.transitions.shape[0]
        self.nb_actions = self.transitions.shape[1]
        #print(self.nb_actions)
        #print(self.transitions.shape)
        #print(self.rewards.shape)

    def set_transitions(self):
        P = np.load("resource_gathering_dynamics.npy")
        return P

    def set_rewards(self):
        R = np.load("resource_gathering_rewards.npy")
        return R