

import math
def sigmoid(h):
    return 1 / (1 + math.exp(-h))
def neural_network(inputs, weights):
    """A simple neural network function that computes the weighted sum of inputs."""
    h = sum(i*w for i,w in zip(inputs,weights))
    activation = sigmoid(h)
    return activation

if __name__ == "__main__":
    inputs = [0.5, 0.25, 0.75]
    weights = [0.4, 0.6, 0.2]
    output = neural_network(inputs, weights)
    print(f"Neural Network Output: {output}")