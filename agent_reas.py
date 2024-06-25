import torch
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0  # Initial randomness
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(17, 256, 3)  # Updated state size of 17
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.causal_graph = self.build_causal_graph()
        self.pos = nx.spring_layout(self.causal_graph)
        self.visualize_causal_graph()
        self.visited_states = set()  # Track visited states for curiosity-driven exploration

    def build_causal_graph(self):
        G = nx.DiGraph()
        G.add_edges_from([
            ('State', 'Action'),
            ('Action', 'Outcome'),
            ('State', 'Outcome')
        ])
        return G

    def visualize_causal_graph(self):
        self.ax.clear()
        nx.draw(self.causal_graph, self.pos, with_labels=True, ax=self.ax, node_size=3000, node_color="skyblue", font_size=12, font_weight="bold", arrows=True)
        plt.title("Causal Graph with Counterfactual Reasoning and Curiosity")
        plt.draw()
        plt.pause(0.001)  # Pause to update the plot

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            # Obstacle proximity
            min(game.head.x, game.w - game.head.x),  # distance to left/right wall
            min(game.head.y, game.h - game.head.y),  # distance to top/bottom wall

            # Tail proximity
            game.snake[1].x < game.head.x,  # tail left
            game.snake[1].x > game.head.x,  # tail right
            game.snake[1].y < game.head.y,  # tail up
            game.snake[1].y > game.head.y  # tail down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def calculate_propensity_scores(self, state):
        """
        Calculate the propensity scores for each action based on the state.
        """
        state_tensor = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(state_tensor), dim=0)
        return probabilities.numpy()

    def counterfactual_reasoning(self, state):
        """
        Use counterfactual reasoning to evaluate potential actions.
        """
        potential_rewards = []
        for action in range(3):
            next_state = self.simulate_action(state, action)
            propensity = self.calculate_propensity_scores(state)[action]
            potential_rewards.append(self.evaluate_state(next_state) * (1 / propensity))
        return potential_rewards

    def simulate_action(self, state, action):
        """
        Simulate the next state given the current state and action.
        """
        simulated_state = state.copy()
        # Simulate the action by updating the state accordingly
        if action == 0:  # Move straight
            if simulated_state[4]:  # Moving left
                simulated_state[0] = simulated_state[1] = simulated_state[2] = 0
                simulated_state[3] = 1
            elif simulated_state[5]:  # Moving right
                simulated_state[0] = simulated_state[1] = simulated_state[2] = 0
                simulated_state[3] = 1
            elif simulated_state[6]:  # Moving up
                simulated_state[0] = simulated_state[1] = simulated_state[2] = 0
                simulated_state[3] = 1
            elif simulated_state[7]:  # Moving down
                simulated_state[0] = simulated_state[1] = simulated_state[2] = 0
                simulated_state[3] = 1
        elif action == 1:  # Move right
            if simulated_state[4]:  # Moving left
                simulated_state[4] = simulated_state[6] = simulated_state[7] = 0
                simulated_state[5] = 1
            elif simulated_state[5]:  # Moving right
                simulated_state[4] = simulated_state[6] = simulated_state[7] = 0
                simulated_state[5] = 1
            elif simulated_state[6]:  # Moving up
                simulated_state[4] = simulated_state[5] = simulated_state[7] = 0
                simulated_state[6] = 1
            elif simulated_state[7]:  # Moving down
                simulated_state[4] = simulated_state[5] = simulated_state[6] = 0
                simulated_state[7] = 1
        elif action == 2:  # Move left
            if simulated_state[4]:  # Moving left
                simulated_state[4] = simulated_state[5] = simulated_state[6] = 0
                simulated_state[7] = 1
            elif simulated_state[5]:  # Moving right
                simulated_state[4] = simulated_state[5] = simulated_state[6] = 0
                simulated_state[7] = 1
            elif simulated_state[6]:  # Moving up
                simulated_state[4] = simulated_state[5] = simulated_state[7] = 0
                simulated_state[6] = 1
            elif simulated_state[7]:  # Moving down
                simulated_state[4] = simulated_state[5] = simulated_state[7] = 0
                simulated_state[6] = 1
        return simulated_state

    def evaluate_state(self, state):
        """
        Evaluate the potential reward of a given state.
        """
        # Example evaluation: Reward based on the proximity to food
        food_distance = np.abs(state[7] - state[11]) + np.abs(state[8] - state[12])
        return -food_distance

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(0.01, self.epsilon * 0.995)  # Decrease epsilon over time
        final_move = [0, 0, 0]
        move = 0

        if random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 2)
        else:
            potential_rewards = self.counterfactual_reasoning(state)
            move = np.argmax(potential_rewards)

        final_move[move] = 1
        return final_move

    def reward_shaping(self, reward, state_old, state_new, action):
        """
        Modify the reward to encourage exploration and prevent turning on itself.
        """
        # Calculate propensity scores
        propensities = self.calculate_propensity_scores(state_old)

        # Use the inverse of the propensity score as a weight
        weight = 1 / propensities[action]

        # Penalize for not moving towards food
        if np.array_equal(state_old[7:11], state_new[7:11]):
            reward -= 2.0  # Higher penalty for not moving towards food

        # Penalize for turning on itself
        if np.array_equal(state_old[:4], state_new[:4]):
            reward -= 2.0  # Higher penalty for repetitive movements

        # Penalize for moving away from food
        if (state_old[7] and state_new[8]) or (state_old[8] and state_new[7]) or \
           (state_old[9] and state_new[10]) or (state_old[10] and state_new[9]):
            reward -= 1.0

        # Additional shaping for moving towards food
        if (state_new[7] and not state_old[7]) or (state_new[8] and not state_old[8]) or \
           (state_new[9] and not state_old[9]) or (state_new[10] and not state_old[10]):
            reward += 2.0  # Higher reward for moving towards food

        # Apply the weight to the reward
        reward *= weight

        # Curiosity-driven exploration: reward for visiting new states
        state_tuple = tuple(state_new)
        if state_tuple not in self.visited_states:
            reward += 1.0
            self.visited_states.add(state_tuple)

        return reward


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        action = np.argmax(final_move)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # reward shaping
        reward = agent.reward_shaping(reward, state_old, state_new, action)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Update the causal graph visualization without clearing
            agent.visualize_causal_graph()

if __name__ == '__main__':
    plt.ion()  # Turn on interactive mode for plotting
    train()
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot
