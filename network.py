import numpy as np
from utils import sigmoid, binary_crossentropy, relu_derivative, relu

# Set seed for reproducibility
np.random.seed(40)


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, epochs, learning_rate, output_size):
        """
        Initialize the neural network with the given parameters.

        Parameters:
            input_size (int): Number of features in the input data.
            hidden_size (int): Number of neurons in the hidden layer.
            epochs (int): Number of iterations to train the model.
            learning_rate (float): Learning rate for gradient descent.
            output_size (int): Number of output neurons (1 for binary classification).
        """
        if input_size <= 0:
            raise ValueError("Input size must be a positive integer.")
        if hidden_size <= 0:
            raise ValueError("Hidden size must be a positive integer.")
        if output_size <= 0:
            raise ValueError("Output size must be a positive integer.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.loss_history = []

        # Initialize weights using He initialization
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)

        # Initialize biases with zeros
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        # Set activation function and its derivative for hidden layer
        self.activation_function = relu
        self.activation_derivative = relu_derivative

    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.

        Parameters:
            X (ndarray): Input data.

        Returns:
            ndarray: Output predictions after sigmoid activation.
        """
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.activation_function(self.hidden_input)

        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.prediction = sigmoid(self.output_input)
        return self.prediction

    def backward_propagation(self, X, y, prediction):
        """
        Perform backpropagation to compute gradients.

        Parameters:
            X (ndarray): Input data.
            y (ndarray): True labels.
            prediction (ndarray): Output from forward propagation.

        Returns:
            Tuple of gradients: (d_w1, d_w2, d_b_hidden, d_b_output)
        """
        # Compute loss gradient w.r.t. output
        d_loss = prediction - y

        # Gradients for output layer weights and bias
        d_w2 = np.dot(self.hidden_output.T, d_loss)
        d_b_output = np.sum(d_loss, axis=0, keepdims=True)

        # Gradients for hidden layer
        dh1 = np.dot(d_loss, self.weights_hidden_output.T)
        dh1 *= self.activation_derivative(self.hidden_input)

        d_w1 = np.dot(X.T, dh1)
        d_b_hidden = np.sum(dh1, axis=0, keepdims=True)

        return d_w1, d_w2, d_b_hidden, d_b_output

    def train(self, X, y, threshold=0.01):
        """
        Train the neural network using gradient descent.

        Parameters:
            X (ndarray): Input training data.
            y (ndarray): True output labels.
            threshold (float): Early stopping threshold for loss.
        """
        for epoch in range(self.epochs):
            # Forward pass
            prediction = self.forward_propagation(X)
            loss = binary_crossentropy(y, prediction)
            mean_loss = np.mean(loss)
            self.loss_history.append(mean_loss)

            # Early stopping if loss is low
            if mean_loss < threshold:
                print(f"Training stopped at epoch {epoch + 1} with loss {mean_loss:.4f}")
                break

            # Backpropagation to compute gradients
            d_w1, d_w2, d_b_hidden, d_b_output = self.backward_propagation(X, y, prediction)

            # Update parameters
            self.weights_hidden_output -= self.learning_rate * d_w2
            self.weights_input_hidden -= self.learning_rate * d_w1
            self.bias_hidden -= self.learning_rate * d_b_hidden
            self.bias_output -= self.learning_rate * d_b_output
