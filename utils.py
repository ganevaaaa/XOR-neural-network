import numpy as np

#Sigmoid activation function
def sigmoid(x):
    """
      Sigmoid activation function.
      Args:
      x : float : input value

      Returns:
      float : output of the sigmoid function
      """
    return 1/(1+np.exp(-x))

# Sigmoid Derivative
def sigmoid_derivative(x):
    """
      Derivative of the sigmoid activation function.
      Args:
      x : float : input value

      Returns:
      float : derivative of the sigmoid function
      """
    return sigmoid(x)*(1-sigmoid(x))

def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-8  # small constant for numerical stability
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))



def binary_crossentropy_derivative(y_true, y_pred):
    """
    Derivative of the Binary Cross-Entropy loss function.

    Args:
    y_true : numpy array : true labels (0 or 1)
    y_pred : numpy array : predicted probabilities (values between 0 and 1)

    Returns:
    numpy array : gradient of the loss with respect to predicted values
    """
    # Clip y_pred to avoid division by zero errors during the derivative calculation
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute the derivative
    grad = (y_pred - y_true) / (y_pred * (1 - y_pred))

    return grad

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)



