import numpy as np
import random


class Qlearning():


    def __init__(self, env, seed=37):
        self.q_table = np.zeros((env.nb_states, env.nb_actions))
        self.params = self.default_parameters()
        self.alpha = self.params['alpha']
        self.discount = self.params['discount']
        self.epsilon = self.params['epsilon']
        self.decaying_rate = self.params['decaying_rate']
        self.env = env
        random.seed(seed)

    @staticmethod
    def default_parameters():
        return dict(
            alpha=0.9,
            discount=0.95,
            epsilon=1,
            decaying_rate=0.001
        )


    def step(self, state, action):
        flat_probs = np.ndarray.flatten(self.env.transitions[state, action])
        next_state = np.random.choice(self.env.nb_states, 1, p=flat_probs)
        reward = self.env.rewards[state, action]
        done = False
        if next_state in self.env.terminal_states:
            done = True
        return next_state, reward, done


    def explore(self):
        return np.random.choice(self.env.nb_actions, 1)


    def learn(self, max_steps, max_eps):
        for i in range(max_eps):
            done = False
            state = self.env.init_state
            for j in range(max_steps):
                if done: break

                if random.uniform(0, 1) < self.epsilon:
                    action = self.explore()
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, done = self.step(state, action)
                #print(state, action, next_state)

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])

                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.discount * next_max)
                self.q_table[state, action] = new_value

                state = next_state


    def softmax_policy(self, beta=0.01):
        assert beta > 0
        policy = np.zeros((self.env.nb_states, self.env.nb_actions))
        for s in range(self.env.nb_states):
            policy[s] = self._softmax_policy(s, beta)
        return policy

    def _softmax_policy(self, s, beta):
        q_values = np.array(self.q_table[s], dtype=np.float128)
        exp = np.exp(beta * q_values)
        return exp / exp.sum()


    def save_policy(self, policy, filename):
        np.save(filename, policy)
