
# XOR Neural Network

This is a simple implementation of a neural network that learns the XOR (exclusive OR) logic function using backpropagation.

## Features
- Fully connected 2-layer neural network  
- ReLU activation in the hidden layer  
- Sigmoid activation in the output layer  
- Binary cross-entropy loss function  
- Manual training using gradient descent  
- Early stopping based on loss threshold  

## Project Structure
The project is organized as follows:
```
├── config.py # Configuration file containing constants like learning rate, epochs, etc.
├── data.py # Data handling (e.g., get_data function for loading XOR data)
├── network.py # Neural network class and its methods
├── utils.py # Utility functions like sigmoid, sigmoid_derivative, etc.
├── main.py # Main script for running the training 
├── tests/ # Directory containing test files
│ └── tests_nn.py # Unit tests for the neural network 
├── README.md # Project description, instructions, etc. 
└── requirements.txt # List of external libraries (e.g., numpy)
```

## Requirements

To install the required Python dependencies, run:

```bash
pip install -r requirements.txt
```
The requirements.txt file includes:

numpy for numerical operations

matplotlib for plotting loss curves


## How to  run the neural network and see the predictions for the XOR function, execute:

```bash
python main.py
```
### Example Output

Once you run the project, you will see output like the following:

```

Training stopped at epoch 120 with loss 0.0099
Input: [0, 0] → Prediction: ~0.0163
Input: [1, 0] → Prediction: ~0.9964
Input: [0, 1] → Prediction: ~0.9967
Input: [1, 1] → Prediction: ~0.0163
```

## Author

You can find me on GitHub: [@ganevaaaa](https://github.com/ganevaaaa)