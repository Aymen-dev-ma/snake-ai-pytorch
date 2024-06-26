import torch
import random
import numpy as np
<<<<<<< HEAD
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import ActorCritic, ActorCriticTrainer
=======
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
>>>>>>> 8b5d73ab71237148cc39b851d4b25df886ca1860
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
<<<<<<< HEAD
LR = 0.01
=======
LR = 0.001
>>>>>>> 8b5d73ab71237148cc39b851d4b25df886ca1860

class Agent:

    def __init__(self):
        self.n_games = 0
<<<<<<< HEAD
        self.epsilon = 0  # randomness
        self.gamma = 0.5  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = ActorCritic(11, 256, 3)
        self.trainer = ActorCriticTrainer(self.model, lr=LR)
=======
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.causal_graph = self.build_causal_graph()
        self.show_causal_graph()

    def build_causal_graph(self):
        # Create a directed graph using networkx
        G = nx.DiGraph()
        # Adding nodes and edges for demonstration purposes
        # In practice, these should be based on domain knowledge or learned from data
        G.add_edges_from([
            ('danger_straight', 'move_left'),
            ('danger_straight', 'move_right'),
            ('danger_right', 'move_left'),
            ('danger_right', 'move_up'),
            ('danger_left', 'move_right'),
            ('danger_left', 'move_up'),
            ('food_left', 'move_left'),
            ('food_right', 'move_right'),
            ('food_up', 'move_up'),
            ('food_down', 'move_down')
        ])
        return G

    def show_causal_graph(self):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.causal_graph)
        nx.draw(self.causal_graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, font_weight="bold", arrows=True)
        plt.title("Causal Graph")
        plt.show(block=True)  # Keeps the plot open

    def query_causal_graph(self, state):
        # Query the causal graph to determine possible actions based on the state
        causal_actions = []
        if state[0]:  # danger straight
            causal_actions.extend(self.causal_graph.successors('danger_straight'))
        if state[1]:  # danger right
            causal_actions.extend(self.causal_graph.successors('danger_right'))
        if state[2]:  # danger left
            causal_actions.extend(self.causal_graph.successors('danger_left'))
        if state[7]:  # food left
            causal_actions.extend(self.causal_graph.successors('food_left'))
        if state[8]:  # food right
            causal_actions.extend(self.causal_graph.successors('food_right'))
        if state[9]:  # food up
            causal_actions.extend(self.causal_graph.successors('food_up'))
        if state[10]:  # food down
            causal_actions.extend(self.causal_graph.successors('food_down'))
        
        # Map action names to indices
        action_map = {'move_left': 0, 'move_right': 1, 'move_up': 2, 'move_down': 3}
        return [action_map[action] for action in set(causal_actions) if action in action_map]

    def counterfactual(self, state, action, outcome):
        # Define the counterfactual reasoning for the snake game
        variables = {
            "danger_straight": state[0],
            "danger_right": state[1],
            "danger_left": state[2],
            "food_left": state[7],
            "food_right": state[8],
            "food_up": state[9],
            "food_down": state[10]
        }
        return variables[outcome]

    def epsilon_greedy(self, state):
        eps = np.random.uniform()
        if eps > self.epsilon:
            return np.argmax(self.model(torch.tensor(state, dtype=torch.float)).detach().numpy())
        
        if self.counterfactual(state, 0, "danger_straight"):
            return 0
        if self.counterfactual(state, 1, "danger_right"):
            return 1
        if self.counterfactual(state, 2, "danger_left"):
            return 2
        
        return random.choice([0, 1, 2])
>>>>>>> 8b5d73ab71237148cc39b851d4b25df886ca1860

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
<<<<<<< HEAD
            game.food.y > game.head.y  # food down
=======
            game.food.y > game.head.y   # food down
>>>>>>> 8b5d73ab71237148cc39b851d4b25df886ca1860
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
<<<<<<< HEAD
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
=======
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
>>>>>>> 8b5d73ab71237148cc39b851d4b25df886ca1860
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
<<<<<<< HEAD
        self.trainer.train_step([state], [action], [reward], [next_state], [done])

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        _, policy_dist = self.model(state)
        policy_dist = policy_dist.detach().numpy()
        action = np.random.choice(len(policy_dist), p=policy_dist)
        final_move = [0, 0, 0]
        final_move[action] = 1
        return final_move

=======
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            move = self.epsilon_greedy(state)
            final_move[move] = 1

        return final_move


>>>>>>> 8b5d73ab71237148cc39b851d4b25df886ca1860
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


if __name__ == '__main__':
    train()
