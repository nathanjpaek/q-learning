import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import logging




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RLSolver:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.filepath = filepath
        
        if 'small' in filepath.lower():
            self.problem_type = 'small'
            self.n_states = 100
            self.n_actions = 4
            self.discount = 0.95
            self.theta = 1e-6  

        elif 'medium' in filepath.lower():
            self.problem_type = 'medium'
            self.n_states = 50000 
            self.n_actions = 7
            self.discount = 1.0
            self.iterations = 50
            self.initial_alpha = 1.0
            self.min_alpha = 0.05
            self.alpha_decay_steps = len(self.data)
            self.terminal_threshold = 475
        else:
            self.problem_type = 'large'
            self.n_states = 302020
            self.n_actions = 9
            self.discount = 0.95
            self.learning_rate = 0.1
            self.episodes = 200
            self.batch_size = 10000
        self.process_data()

    def process_data(self):
        self.transitions = defaultdict(list)
        self.rewards = defaultdict(list)
        self.terminal_states = set()
        
        for _, row in self.data.iterrows():
            s = int(row['s']) - 1
            a = int(row['a']) - 1
            r = float(row['r'])
            sp = int(row['sp']) - 1
            
            self.transitions[(s, a)].append((sp, r))
            
            if self.problem_type == 'medium':
                pos = s % 500
                vel = s // 500
                if pos + 0.3 * vel > self.terminal_threshold:
                    self.terminal_states.add(s)

    def solve_small(self):
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)
        
        for iteration in tqdm(range(1000)):
            delta = 0
            for s in range(self.n_states):
                v = V[s]
                values = []
                
                for a in range(self.n_actions):
                    if (s, a) in self.transitions:
                        trans = self.transitions[(s, a)]
                        exp_value = 0
                        for sp, r in trans:
                            p = 1.0 / len(trans)
                            exp_value += p * (r + self.discount * V[sp])
                        values.append(exp_value)
                
                if values:
                    V[s] = max(values)
                    delta = max(delta, abs(v - V[s]))
            
            if delta < self.theta:
                break
        
        for s in range(self.n_states):
            values = []
            for a in range(self.n_actions):
                if (s, a) in self.transitions:
                    trans = self.transitions[(s, a)]
                    exp_value = 0
                    for sp, r in trans:
                        p = 1.0 / len(trans)
                        exp_value += p * (r + self.discount * V[sp])
                    values.append((exp_value, a))
            
            if values:
                policy[s] = max(values)[1]
            else:
                policy[s] = s % self.n_actions
        
        self.policy = policy + 1

    def solve_medium(self):
        Q_values = np.zeros((self.n_states, self.n_actions))
        policy = np.zeros(self.n_states)
        gamma = self.discount 
        alpha_initial = self.initial_alpha
        alpha_min = self.min_alpha
        alpha_decay_steps = self.alpha_decay_steps

        for iteration in range(self.iterations):
            step_count = 0
            for _, row in self.data.iterrows():
                state = int(row['s']) - 1
                action = int(row['a']) - 1
                reward = float(row['r'])
                next_state = int(row['sp']) - 1

                # update lr to decay over steps
                alpha = alpha_initial - (alpha_initial - alpha_min) * step_count / alpha_decay_steps

                next_best_action = np.argmax(Q_values[next_state])
                Q_values[state, action] += alpha * (
                    reward + gamma * Q_values[next_state, next_best_action] - Q_values[state, action]
                )

                step_count += 1

        max_reward_actions = {}
        for _, row in self.data.iterrows():
            state = int(row['s']) - 1
            action = int(row['a']) - 1
            reward = float(row['r'])

            if state not in max_reward_actions or reward > max_reward_actions[state][1]:
                max_reward_actions[state] = (action + 1, reward)

        for state in range(self.n_states):
            if np.any(Q_values[state] != 0):
                policy[state] = np.argmax(Q_values[state]) + 1
            else:
                # if state was not visited, assign action with the max observed reward / a random action
                random_action = np.random.randint(1, self.n_actions + 1)
                policy[state] = max_reward_actions.get(state, (random_action, 0))[0]

        self.policy = policy


    def solve_q_learning(self):
        Q = defaultdict(lambda: np.zeros(self.n_actions))
        
        for episode in tqdm(range(self.episodes), desc='Training Episodes'):
            for _, row in self.data.iterrows():
                s = int(row['s']) - 1
                a = int(row['a']) - 1
                r = float(row['r'])
                sp = int(row['sp']) - 1
                
                max_Q_sp = np.max(Q[sp]) if sp in Q else 0
                td_target = r + self.discount * max_Q_sp
                td_error = td_target - Q[s][a]
                Q[s][a] += self.learning_rate * td_error
        
        policy = np.zeros(self.n_states, dtype=int)
        for s in tqdm(range(self.n_states), desc='Extracting Policy'):
            if s in Q and np.any(Q[s] != 0):
                policy[s] = np.argmax(Q[s])
            else:
                policy[s] = 0
        
        self.policy = policy + 1 

    def solve(self):
        if self.problem_type == 'small':
            self.solve_small()
        elif self.problem_type == 'medium':
            self.solve_medium()
        else:
            self.solve_q_learning()

    def save_policy(self):
        output_file = os.path.splitext(self.filepath)[0] + '.policy'
        np.savetxt(output_file, self.policy, fmt='%d')

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_file')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    try:
        solver = RLSolver(args.input_file)
        solver.solve()
        solver.save_policy()
    except Exception as e:
        logging.error(f"error: {str(e)}")
        raise

if __name__ == "__main__":
    main()