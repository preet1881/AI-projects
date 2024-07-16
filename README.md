# Deep Q-Network (DQN) Implementation for Lunar Lander

This repository contains an implementation of the Deep Q-Network (DQN) algorithm applied to the Lunar Lander environment from OpenAI's Gymnasium. The DQN algorithm is a reinforcement learning method that combines Q-learning with deep neural networks to handle environments with large state spaces.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Explanation](#algorithm-explanation)
- [Code Structure](#code-structure)
- [Libraries and Modules](#libraries-and-modules)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this implementation, you'll need to have Python installed along with the necessary libraries. You can install the required libraries using pip.

```bash
pip install gymnasium
pip install "gymnasium[atari, accept-rom-license]"
apt-get install -y swig
pip install gymnasium[box2d]
pip install torch numpy imageio
```

## Usage

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/lunar-lander-dqn.git
cd lunar-lander-dqn
```

2. **Run the training script:**

```python
python train.py
```

3. **Visualize the trained agent:**

```python
python visualize.py
```

## Algorithm Explanation

### Deep Q-Network (DQN)

DQN is a model-free, off-policy algorithm that uses a neural network to approximate the Q-value function. The main components of DQN are:

1. **Experience Replay**: Stores the agent's experiences and samples random batches for training, which helps to break the correlation between consecutive experiences and stabilize learning.
2. **Target Network**: A separate network used to compute target Q-values, which is updated less frequently to improve stability.
3. **Epsilon-Greedy Policy**: Balances exploration and exploitation by selecting random actions with probability epsilon and the best-known action with probability 1-epsilon.

### Neural Network Architecture

The neural network consists of three fully connected (linear) layers with ReLU activation functions between them. The architecture is as follows:

- **Input Layer**: Takes the state as input.
- **Hidden Layers**: Two hidden layers with 64 units each, followed by ReLU activations.
- **Output Layer**: Produces action values for each possible action.

## Code Structure

- **train.py**: Contains the main training loop and the DQN agent implementation.
- **visualize.py**: Contains the code for visualizing the trained agent.
- **network.py**: Defines the neural network architecture.
- **replay_memory.py**: Implements the experience replay buffer.
- **agent.py**: Implements the DQN agent.

## Libraries and Modules

### PyTorch

PyTorch is used for building and training the neural network. Key modules include:

- **torch**: The main PyTorch library.
- **torch.nn**: For defining the neural network layers.
- **torch.optim**: For optimization algorithms (Adam).
- **torch.nn.functional (F)**: For activation functions and loss computation.

### Python Standard Libraries

- **os**: For operating system-dependent functionality.
- **random**: For random number generation.
- **collections**: For specialized container datatypes like deque and namedtuple.
- **numpy**: For numerical operations on arrays.
- **base64**: For encoding binary data.
- **glob**: For file path matching.
- **io**: For stream handling.

### Gymnasium

Gymnasium is used for the Lunar Lander environment and provides the standard API for interacting with the environment.

- **gymnasium**: The main library for creating and interacting with environments.
- **gym.wrappers.monitoring.video_recorder**: For recording videos of the agent's performance.

## Results

During training, the average score per episode is printed to monitor the agent's performance. The trained agent can be visualized by running the `visualize.py` script, which records a video of the agent interacting with the environment.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README further based on your specific project details and preferences.
