from data import get_data
from config import LEARNING_RATE, EPOCHS, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from network import NeuralNetwork
import matplotlib.pyplot as plt

X, Y = get_data()

neural_Network = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, EPOCHS, LEARNING_RATE, OUTPUT_SIZE)
neural_Network.train(X, Y)

for x in X:
        prediction = neural_Network.forward_propagation(x.reshape(1, -1))
        approx = prediction.item()  # Safe scalar extraction
        print(f"Input: {x.tolist()} → Prediction: ~{approx:.4f}")

# Plot only every 100th epoch for clarity
plt.plot(neural_Network.loss_history[::100])
plt.title("Loss over Epochs")
plt.xlabel("Epoch (every 1000th)")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.show()
