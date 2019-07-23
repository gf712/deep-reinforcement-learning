import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.4, gamma=0.95, epsilon=1.0, epsilon_decay=2000, min_epsilon=0.005):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def select_action(self, env, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(np.arange(self.nA), p=self._get_probabilities(state)) if state in self.Q \
               else env.action_space.sample()
    
    def _get_probabilities(self, state):
        
        p = np.full(self.nA, self.epsilon / self.nA)
        p[np.argmax(self.Q[state])] = 1 - self.epsilon + self.epsilon / self.nA
        return p
    
    def update_epsilon(self, episode_number):
        if episode_number % self.epsilon_decay == 0:
            self.epsilon = self._epsilon
        else:
            self.epsilon = max((self.epsilon/2) * (np.cos(episode_number/self.epsilon_decay*np.pi)+1), self.min_epsilon)
#         self.epsilon = max(self.epsilon * np.exp(-episode_number*self.epsilon_decay),
#                            self.min_epsilon)
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Expected SARSA
#         if not done:
#             self.Q[state][action] = self.Q[state][action] + \
#                                     self.alpha * (reward + self.gamma * sum(p * self.Q[next_state][i] for i, p in \
#                                                                        enumerate(self._get_probabilities(next_state))) \
#                                                   - self.Q[state][action])
#         else:
#             self.Q[state][action] = self.Q[state][action] + self.alpha * (reward - self.Q[state][action])
        # Q-learning
        if not done:
            self.Q[state][action] = self.Q[state][action] + \
                                    self.alpha * (reward + self.gamma * max(self.Q[next_state]) \
                                                  - self.Q[state][action])
        else:
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward - self.Q[state][action])
            
        