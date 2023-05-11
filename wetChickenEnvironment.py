import wetChicken
import spibb_utils

class Chicken():

    def __init__(self):
        self.chicken = wetChicken.WetChicken(5, 5, 3.5, 3)
        self.transitions = self.chicken.get_transition_function()
        self.rewards = spibb_utils.get_reward_model(self.transitions, self.chicken.get_reward_function())
        self.nb_states = self.chicken.get_nb_states()
        self.nb_actions = self.chicken.get_nb_actions()
        self.init_state = self.chicken.get_state_int()
        self.terminal_states = []