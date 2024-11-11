import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

class RLSolver:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.filepath = filepath
        
        # Determine problem type and setup parameters
        if 'small' in filepath:
            self.problem_type = 'small'
            self.n_states = 100  # 10x10 grid
            self.n_actions = 4   # left, right, up, down
            self.discount = 0.95
            self.theta = 1e-5  # Convergence threshold for value iteration
        elif 'medium' in filepath:
            self.problem_type = 'medium'
            self.n_states = 50000  # 500 positions × 100 velocities
            self.n_actions = 7     # Acceleration values
            self.discount = 1.0
            self.learning_rate = 0.1
            self.episodes = 10  # Reduced for efficiency
        else:  # large
            self.problem_type = 'large'
            self.n_states = 10101010  # Correct number of states
            self.n_actions = 125      # Correct number of actions
            self.discount = 0.95
            self.learning_rate = 0.1
            self.episodes = 1  # Only one pass due to large size

        # Initialize transition and reward dictionaries
        self.transitions = defaultdict(lambda: defaultdict(list))
        self.rewards = defaultdict(lambda: defaultdict(float))
        self.process_data()

    def process_data(self):
        """Process the CSV data into transitions and rewards"""
        for _, row in self.data.iterrows():
            # Adjust indices to be 0-based
            s = int(row['s']) - 1
            a = int(row['a']) - 1
            r = float(row['r'])
            sp = int(row['sp']) - 1
            self.transitions[s][a].append(sp)
            self.rewards[s][a] = r  # Assuming deterministic rewards

    def solve_small(self):
        """Value Iteration for the small problem"""
        num_states = self.n_states
        num_actions = self.n_actions
        discount_factor = self.discount
        theta = self.theta

        U = np.zeros(num_states)  # State-value function
        policy = np.zeros(num_states, dtype=int)

        while True:
            delta = 0
            for s in range(num_states):
                v = U[s]
                action_values = []
                for a in range(num_actions):
                    if a in self.transitions[s]:
                        sp_list = self.transitions[s][a]
                        r = self.rewards[s][a]
                        U_sp = np.mean([U[sp] for sp in sp_list])
                        action_values.append(r + discount_factor * U_sp)
                if action_values:
                    U[s] = max(action_values)
                    delta = max(delta, abs(v - U[s]))
            if delta < theta:
                break

        # Derive policy
        for s in range(num_states):
            action_values = []
            for a in range(num_actions):
                if a in self.transitions[s]:
                    sp_list = self.transitions[s][a]
                    r = self.rewards[s][a]
                    U_sp = np.mean([U[sp] for sp in sp_list])
                    action_values.append((r + discount_factor * U_sp, a))
            if action_values:
                policy[s] = max(action_values)[1]
            else:
                policy[s] = 0  # Default action

        self.policy = policy + 1  # Convert to 1-based indexing

    def solve_q_learning(self):
        """Q-learning algorithm for medium and large problems"""
        # Use a dictionary to store Q-values for efficiency
        Q = defaultdict(lambda: np.zeros(self.n_actions))
        epsilon = 0.1  # Epsilon for epsilon-greedy policy

        for episode in tqdm(range(self.episodes), desc='Training Episodes'):
            # Shuffle data for each episode (optional for better convergence)
            data_shuffled = self.data.sample(frac=1).reset_index(drop=True)
            for _, row in data_shuffled.iterrows():
                s = int(row['s']) - 1
                a = int(row['a']) - 1
                r = float(row['r'])
                sp = int(row['sp']) - 1

                # Epsilon-greedy action selection
                if np.random.rand() < epsilon:
                    a_selected = np.random.randint(self.n_actions)
                else:
                    a_selected = np.argmax(Q[s])

                # Q-learning update
                max_Q_sp = np.max(Q[sp]) if sp in Q else 0
                td_target = r + self.discount * max_Q_sp
                td_error = td_target - Q[s][a]
                Q[s][a] += self.learning_rate * td_error

        # Derive policy
        policy = np.zeros(self.n_states, dtype=int)

        for s in tqdm(range(self.n_states), desc='Extracting Policy'):
            if s in Q and np.any(Q[s] != 0):
                policy[s] = np.argmax(Q[s])
            else:
                policy[s] = 0  # Default action

        self.policy = policy + 1  # Convert to 1-based indexing

    def solve(self):
        if self.problem_type == 'small':
            self.solve_small()
        else:
            self.solve_q_learning()

    def save_policy(self):
        """Save the policy to a file"""
        output_file = self.filepath.replace('.csv', '.policy')
        np.savetxt(output_file, self.policy, fmt='%d')

def solve_problem(filepath):
    solver = RLSolver(filepath)
    solver.solve()
    solver.save_policy()

if __name__ == "__main__":
    # Solve all three problems
    for file in ['small.csv', 'medium.csv', 'large.csv']:
        print(f"\nSolving {file}...")
        solve_problem(file)
