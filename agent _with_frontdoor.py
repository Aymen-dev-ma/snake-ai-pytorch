import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

class SCM:
    def predict_next_state(self, state, action):
        head_x, head_y = state['head_x'], state['head_y']
        next_head = Point(head_x, head_y)

        if action == 0:  # Straight
            if state['direction'] == Direction.RIGHT:
                next_head = Point(head_x + 20, head_y)
            elif state['direction'] == Direction.LEFT:
                next_head = Point(head_x - 20, head_y)
            elif state['direction'] == Direction.UP:
                next_head = Point(head_x, head_y - 20)
            elif state['direction'] == Direction.DOWN:
                next_head = Point(head_x, head_y + 20)
        elif action == 1:  # Right turn
            if state['direction'] == Direction.RIGHT:
                next_head = Point(head_x, head_y + 20)
            elif state['direction'] == Direction.LEFT:
                next_head = Point(head_x, head_y - 20)
            elif state['direction'] == Direction.UP:
                next_head = Point(head_x + 20, head_y)
            elif state['direction'] == Direction.DOWN:
                next_head = Point(head_x - 20, head_y)
        elif action == 2:  # Left turn
            if state['direction'] == Direction.RIGHT:
                next_head = Point(head_x, head_y - 20)
            elif state['direction'] == Direction.LEFT:
                next_head = Point(head_x, head_y + 20)
            elif state['direction'] == Direction.UP:
                next_head = Point(head_x - 20, head_y)
            elif state['direction'] == Direction.DOWN:
                next_head = Point(head_x + 20, head_y)
        return {
            'head_x': next_head.x, 
            'head_y': next_head.y, 
            'direction': state['direction'], 
            'food_position': state['food_position']
        }
    def calculate_reward(self, next_state, game):
        head = Point(next_state['head_x'], next_state['head_y'])
        food = next_state['food_position']

        # Calculate reward
        if game.is_collision(head):
            return -10
        elif head == food:
            return 10
        else:
            return 0
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=100000)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
        self.scm = SCM()

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

        state = {
            'head_x': head.x,
            'head_y': head.y,
            'direction': game.direction,
            'food_position': game.food,
            'state_array': [
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
        }
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state['state_array'], dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        # Frontdoor adjustment: Use SCM to predict next state and adjust action selection
        next_state = self.scm.predict_next_state(state, move)
        adjusted_reward = self.scm.calculate_reward(next_state, game)
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old, game)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory (replay)
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

if __name__ == "__main__":
    train()
