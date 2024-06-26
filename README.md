# Snake AI with Causal Inference

This project implements a Snake AI using causal inference techniques to enhance decision-making and reward shaping. The AI uses both frontdoor and backdoor adjustments to optimize its policy.

## Overview

The goal is to train a Snake AI that maximizes its cumulative reward by eating food and avoiding collisions. The AI incorporates causal inference to better understand the impact of its actions and adjust its behavior accordingly.

## Key Components

1. **Structural Causal Model (SCM)**:
    - **Nodes**: `State_t`, `Action_t`, `FoodPlacement_t`, `Reward_t`, `State_t+1`, `Collision_t`
    - **Edges**: Relationships between these nodes to capture causal dependencies.

2. **Frontdoor Adjustment**:
    - Used for action selection to account for the indirect effect of actions through intermediate variables.
    - Adjusts action probabilities based on predicted future states.

3. **Backdoor Adjustment**:
    - Used for reward shaping to account for confounding variables (e.g., food placement).
    - Adjusts rewards based on the predicted next state and potential collisions.

## Code Structure

- `agent.py`: Defines the Snake AI agent, including action selection and reward shaping using causal inference.
- `model.py`: Defines the neural network model for Q-learning.
- `game.py`: Implements the Snake game environment.
- `scm.py`: Implements the structural causal model and causal inference techniques.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- NetworkX
- Pygame

## Installation

Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Train the Snake AI:
    ```sh
    python agent.py
    ```

2. Observe the training progress and performance metrics.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.