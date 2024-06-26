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
        self.model = Linear_QNet(11, 256, 3)  # State size of 11
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.causal_graph = self.build_causal_graph()
        self.pos = nx.spring_layout(self.causal_graph)
        self.visualize_causal_graph()

    def build_causal_graph(self):
        G = nx.DiGraph()
        G.add_edges_from([
            ('Obstacle Proximity', 'Action'),
            ('Action', 'Outcome'),
            ('Obstacle Proximity', 'Outcome')
        ])
        return G

    def visualize_causal_graph(self):
        nx.draw(self.causal_graph, self.pos, with_labels=True, ax=self.ax, node_size=3000, node_color="skyblue", font_size=12, font_weight="bold", arrows=True)
        plt.title("Causal Graph with Backdoor Adjustment")
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
            game.food.y > game.head.y  # food down
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

    def backdoor_adjustment(self, state):
        """
        Backdoor adjustment to add reasoning based on obstacle proximity.
        """
        obstacle_proximity = [state[-2], state[-1]]  # Last two elements in state represent obstacle proximity
        # Use the backdoor adjustment to influence action choices
        # For example, if the agent is close to an obstacle, it should avoid it
        if obstacle_proximity[0] < 20 or obstacle_proximity[1] < 20:
            return -1  # High risk, avoid this action
        return 0  # Safe action

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(0.01, self.epsilon - 0.001)  # Decrease epsilon over time
        final_move = [0, 0, 0]

        if random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()

        causal_adjustment = self.backdoor_adjustment(state)
        if causal_adjustment == -1:
            move = (move + 1) % 3  # Change action to avoid obstacle

        final_move[move] = 1
        return final_move

    def reward_shaping(self, reward, state_old, state_new):
        """
        Modify the reward to encourage exploration and prevent turning on itself.
        """
        # Penalize for not moving towards food
        if np.array_equal(state_old[7:11], state_new[7:11]):
            reward -= 0.2

        # Penalize for turning on itself
        if np.array_equal(state_old[:4], state_new[:4]):
            reward -= 0.2

        # Additional shaping for moving towards food
        if (state_new[7] and not state_old[7]) or (state_new[8] and not state_old[8]) or \
           (state_new[9] and not state_old[9]) or (state_new[10] and not state_old[10]):
            reward += 0.5

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

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # reward shaping
        reward = agent.reward_shaping(reward, state_old, state_new)

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
