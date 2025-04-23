import unittest
import numpy as np
from network import NeuralNetwork  # Import your network class

class TestNeuralNetwork(unittest.TestCase):
    """
    Unit tests for the NeuralNetwork class to ensure correct functionality.
    Tests include forward propagation, loss reduction, and input validation.

    Methods:
    setUp() -> Sets up the test environment and initializes the neural network.
    test_forward_propagation() -> Tests the forward propagation method.
    test_loss_reduction() -> Tests that the loss decreases during training.
    test_invalid_input() -> Tests that invalid input raises an error.
    test_invalid_data() -> Tests shape consistency during backpropagation.
    test_training_stops_early() -> Tests that training stops early when the loss is sufficiently low.
    """

    def setUp(self):
        """
        Sets up the test data (XOR inputs and outputs).
        This will be used in all test cases.
        """
        # Define XOR inputs and expected outputs
        self.X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # Input for XOR
        self.Y = np.array([[0], [1], [1], [0]])  # Expected XOR output

        # Initialize the neural network
        self.nn = NeuralNetwork(input_size=2, hidden_size=4, epochs=100000, learning_rate=0.5, output_size=1)
        self.nn.train(self.X, self.Y)

    def test_forward_propagation(self):
        """
        Tests that the forward propagation method returns the correct shape of output.
        """
        prediction = self.nn.forward_propagation(self.X)
        self.assertEqual(prediction.shape, (4, 1), "Prediction shape should be (4, 1).")

    def test_loss_reduction(self):
        """
        Tests that the loss decreases during training.
        """
        self.nn.train(self.X, self.Y)
        initial_loss = self.nn.loss_history[0]
        final_loss = self.nn.loss_history[-1]
        self.assertLess(final_loss, initial_loss, "Loss should decrease during training.")

    def test_invalid_input(self):
        """
        Tests that the forward propagation raises an error for invalid input shapes.
        """
        with self.assertRaises(ValueError):
            self.nn.forward_propagation(np.array([[0]]))  # Invalid input shape

    def test_invalid_data(self):
        """
        Tests shape consistency during backward propagation.
        """
        with self.assertRaises(ValueError):
            self.nn.backward_propagation(self.X, np.array([[0]]), np.array([[1]]))  # Mismatched shapes

    def test_training_stops_early(self):
        """
        Tests that training stops early when the loss reaches a sufficiently low value.
        """
        self.nn.train(self.X, self.Y, threshold=0.05)
        self.assertTrue(len(self.nn.loss_history) < self.nn.epochs, "Training should stop early when the loss is sufficiently low.")

    def test_prediction(self):
        """
        Test predictions for each XOR input.
        """
        for i, sample in enumerate(self.X):
            prediction = self.nn.forward_propagation(sample.reshape(1, -1))[0, 0]
            expected_output = self.Y[i][0]
            # The prediction should be close to the expected output (allowing for a small error)
            self.assertAlmostEqual(prediction, expected_output, delta=0.05)

    def test_loss_decreases(self):
        initial_loss = self.nn.loss_history[0]
        final_loss = self.nn.loss_history[-1]
        self.assertLess(final_loss, initial_loss)

    def test_correct_output(self):
        """
        Test if the network outputs close to 0 or 1 for XOR inputs.
        """
        predictions = [self.nn.forward_propagation(sample.reshape(1, -1))[0, 0] for sample in self.X]
        for i, pred in enumerate(predictions):
            if self.Y[i] == 0:
            # Allowing a range of tolerance for values close to 0
                self.assertLess(pred, 0.5 + 0.1, f"Expected output close to 0, got {pred}")
        else:
            # Allowing a range of tolerance for values close to 1
                self.assertGreater(pred, 0.5 - 0.1, f"Expected output close to 1, got {pred}")

    def test_training_stops_on_threshold(self):
        """
        Check if the training stops when the loss is below the threshold.
        """

        # Ensure the training was completed and loss history is populated
        if not self.nn.loss_history:
            self.fail(f"Training did not run or loss history is empty. Loss history: {self.nn.loss_history}")

        final_loss = self.nn.loss_history[-1]
        print(f"Final loss after training: {final_loss}")
        self.assertLess(final_loss, 0.01)


if __name__ == "__main__":
    unittest.main()
