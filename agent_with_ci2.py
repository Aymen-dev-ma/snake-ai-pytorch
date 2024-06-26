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
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 4)  # Changed output size to 4 to accommodate all actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.causal_graph = self.build_causal_graph()
        self.pos = nx.spring_layout(self.causal_graph)
        self.visualize_causal_graph()

    def build_causal_graph(self):
        # Create a directed graph using networkx
        G = nx.DiGraph()
        # Adding nodes and edges for the rule-based reasoning
        G.add_edges_from([
            ('danger_straight', 'move_left'),
            ('danger_straight', 'move_right'),
            ('danger_right', 'move_left'),
            ('danger_right', 'move_up'),
            ('danger_left', 'move_right'),
            ('danger_left', 'move_up'),
            ('no_danger', 'move_towards_food'),
            ('food_left', 'move_left'),
            ('food_right', 'move_right'),
            ('food_up', 'move_up'),
            ('food_down', 'move_down')
        ])
        return G

    def visualize_causal_graph(self):
        self.ax.clear()
        nx.draw(self.causal_graph, self.pos, with_labels=True, ax=self.ax, node_size=3000, node_color="skyblue", font_size=12, font_weight="bold", arrows=True)
        plt.title("Causal Graph for Rule-Based Reasoning")
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
            game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def rule_based_causal_reasoning(self, state):
        """
        Define a set of rules to determine the best action based on the current state.
        """
        danger_straight = state[0]
        danger_right = state[1]
        danger_left = state[2]
        food_left = state[7]
        food_right = state[8]
        food_up = state[9]
        food_down = state[10]

        # Prioritize avoiding danger
        if danger_straight:
            if not danger_right:
                return 1  # Move right
            elif not danger_left:
                return 0  # Move left
            elif not food_up:
                return 2  # Move up
            else:
                return 3  # Move down
        if danger_right and not danger_left:
            return 0  # Move left
        if danger_left and not danger_right:
            return 1  # Move right

        # If no immediate danger, prioritize moving towards the food
        if food_left:
            return 0  # Move left
        if food_right:
            return 1  # Move right
        if food_up:
            return 2  # Move up
        if food_down:
            return 3  # Move down

        # If no clear decision, choose a random safe move
        safe_moves = [0, 1, 2, 3]
        safe_moves = [move for move in safe_moves if not self.simulate_danger(state, move)]
        if safe_moves:
            return random.choice(safe_moves)
        
        return random.randint(0, 3)  # Fallback to a random move

    def simulate_danger(self, state, action):
        """
        Simulate the result of taking a given action and return if it results in danger.
        """
        if action == 0:  # Move left
            return state[2]
        if action == 1:  # Move right
            return state[1]
        if action == 2:  # Move up
            return state[0]
        if action == 3:  # Move down
            return state[0]  # Assuming moving down is as dangerous as moving straight

        return False

    def epsilon_greedy(self, state):
        eps = np.random.uniform()
        if eps > self.epsilon:
            return np.argmax(self.model(torch.tensor(state, dtype=torch.float)).detach().numpy())
        return self.rule_based_causal_reasoning(state)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            move = self.epsilon_greedy(state)
            final_move[move] = 1

        return final_move


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

            # Update the causal graph visualization
            agent.visualize_causal_graph()

if __name__ == '__main__':
    plt.ion()  # Turn on interactive mode for plotting
    train()
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot
