# Snake Game AI using Q-Learning with Causal Inference
import random
import numpy as np
import torch
from collections import deque
from game import SnakeGameAI, Direction, Point
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

def counterfactual(state, action, outcome, game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)
    
    food_left = game.food.x < head.x
    food_right = game.food.x > head.x
    food_up = game.food.y < head.y
    food_down = game.food.y > head.y
    
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    variables = {
        "goal": False,
        "danger": False,
        "direction": False
    }
    
    # Check for food
    if (action == 0 and food_left) or (action == 1 and food_right) or (action == 2 and food_up) or (action == 3 and food_down):
        variables["goal"] = True
    
    # Check for danger
    if (action == 0 and game.is_collision(point_l)) or (action == 1 and game.is_collision(point_r)) or (action == 2 and game.is_collision(point_u)) or (action == 3 and game.is_collision(point_d)):
        variables["danger"] = True
    
    # Check for safe direction
    if not variables["danger"]:
        variables["direction"] = True
    
    return variables[outcome]

class QLearningCausal:
    def __init__(self, game, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.game = game
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}
    
    def get_current_value(self):
        return self.epsilon
    
    def epsilon_greedy(self, state):
        eps = np.random.uniform()
        if eps > self.get_current_value():
            return np.argmax(self.Q.get(state, [0, 0, 0, 0]))
        
        if counterfactual(state, 0, "goal", self.game):
            return 0
        if counterfactual(state, 1, "goal", self.game):
            return 1
        if counterfactual(state, 2, "goal", self.game):
            return 2
        if counterfactual(state, 3, "goal", self.game):
            return 3
        
        safe_actions = [a for a in range(4) if counterfactual(state, a, "direction", self.game)]
        if safe_actions:
            return np.random.choice(safe_actions)
        
        return np.random.choice(4)
    
    def train(self):
        for episode in range(self.episodes):
            self.game.reset()
            state = tuple(self.game.get_state())
            done = False
            while not done:
                action = self.epsilon_greedy(state)
                reward, done, score = self.game.play_step(action)
                new_state = tuple(self.game.get_state())
                best_next_action = np.argmax(self.Q.get(new_state, [0, 0, 0, 0]))
                self.Q[state][action] = self.Q.get(state, [0, 0, 0, 0])[action] + self.alpha * (reward + self.gamma * self.Q.get(new_state, [0, 0, 0, 0])[best_next_action] - self.Q.get(state, [0, 0, 0, 0])[action])
                state = new_state

    def test(self):
        self.game.reset()
        state = tuple(self.game.get_state())
        done = False
        while not done:
            action = np.argmax(self.Q.get(state, [0, 0, 0, 0]))
            _, done, _ = self.game.play_step(action)
            state = tuple(self.game.get_state())

def main():
    game = SnakeGameAI()
    episodes = 1000
    q_learning_causal = QLearningCausal(game, episodes=episodes)
    q_learning_causal.train()
    q_learning_causal.test()

if __name__ == '__main__':
    main()
