# ann-tictactoe

## Abstract
This project uses a neural network to play tic-tac-toe. The neural network is a feed-forward type and uses backpropogation to learn.

## Introduction
Previously in this class, we have created an AI to play tic-tac-toe using the minimax algorithm with alpha-beta pruning. This project is to apply the concept of neural networks and deep learning to create an AI for tic-tac-toe. The goal here is to create a console-based tic-tac-toe game with the same outward appearance of the previous project, however now a neural network is the brain behind the AI. The minimax algorithm is used to determine best move options and generate test cases, which are then fed to the neural network to develop and learn the best moves in the game.

## Problem Definition
Use a forward-feed type artificial neural network (ANN) to play the tic-tac-toe game. The ANN should learn which moves to make in the game through deep learning using a backpropogation algorithm. In tic-tac-toe, two players take turns placing marks down on a 3x3 grid, until one player has made three marks in a row, either in a horizontal, vertical, or diagonal fashion. 
The ANN used should have an input layer accepting a game state of the current game board as well as the current perspective (whether to find the best move for 'o' or 'x' player). An output layer of the ANN should have one node, which describes the best tile to move to. Any number of hidden layer can be used, with variable numbers of nodes per layer. However, one hidden layer with several (1-10) nodes ought to be optimal.

## Methods
For this class, a library was suggested for use which implemented the various matrix methods used in an ANN. I wanted to implement my network from scratch without the use of any external libraries. As such, its necessary for this program to define methods for forward-feeding input data, backpropogation of error, updating node weights to allow the network to learn, activation of a node, and derivative of the activation function used for gradient calculation. Furthermore, effective generation of input data must be done to properly train the network.

The forward-feeding algorithm works through the network layer by layer and calculates the activation and output of each node. Activation of any one neuron is equal to the sum of the output of each input node multiplied by the weight of the link between that node and the neuron, plus the bias of that node. The output of the node found by passing the activation value to the transfer function.

The transfer function currently in use for this program is the sigmoid function, which is defined as f(x) = 1 / (1 + e^x) , whereas f'(x) = f(x) * (1 - f(x).

Back propogation is used to calculate the gradient and error of the network, which is necessary to be able to teach the network better weights to use between nodes. The algorithm starts at the output layer of the ANN, and works backwords through each layer. Delta is used as an indicator of overall error of a neuron. The delta of each neuron in the output layer is calculated as the expected output of the neuron minus the actual output of the neuron, multiplied by the transfer derivative of the neuron's output. Each inner layer then uses slightly different formula for error calculation. Each neuron's delta of inner layers is defined as the sum of the delta of each neuron in the next layer multiplied by the weight of the neuron in that layer. That entire value is multiplied by the transfer derivative of the current neuron to calculate the final delta value.
	
After backpropogation determines delta error of nodes in a network, these error values can be used to correct the weights between nodes. The learning method 'update_weights' goes through each layer, starting with the first hidden layer, and starts with the input passed into the network to find the network's current output values. For each layer in this method, in each neuron in that layer, the neuron's weights are added to by a small sum, taking into account the neuron's delta value. The additive value for one weight in a neuron is summed as the delta value of the neuron, multiplied by the learning rate, multiplied by the neuron's output value, multiplied by x, for each output value x in the previous layer of the network. The neuron's bias is similarly added to with the value of the neuron's delta value multiplied by the learning rate. The output of each node in a layer is used as the input for the next layer.

The learning rate used in this algorithm is 0.5, and acts as a coefficient for how fast neuron weights are updated. 
In this ANN, an input layer of size 10 is used. The first 9 represent the values (empty, 'x' or 'o') of each of the nine tiles in the game board, in row-major order. The last input node represents the perspective to evaluate the game under; in other words, whether the ANN will place an 'o' or an 'x'. The previous tic-tac-toe project is used to generate various game states, by playing a variable number of games, while pitching the minimax algorithm against a random move generator. For every turn, the associated expected output is the best move as determined by the minimax algorithm. To teach the ANN to play as both players, for each game, the point of view (POV) of whether the NN is 'x' or 'o', as well as whether the minimax algorithm plays as 'x' or 'o', is randomly decided. 

## Results
This ANN never was able to learn how to play tic-tac-toe effectively. While the delta error values converged upon zero as expected, the output value of the ANN also converged upon one value, irregardless of the supplied input. This also depended upon the transfer function used. With the sigmoid function, currently in use, output converges to 0. SoftPlus caused the output to converge upon ~0.5. Sine caused the output to never converge. Tanh caused the network to converge upon -1.0. These functions and their derivatives were found at the wikipedia page as listed in the references section of this writeup.
	
A test network can be quickly created and evaluated with x nodes in the hidden layer with the command:
~~~
python3 -c 'import nn;nn.test_network(x)'
~~~
Optionally, parameters for the number of training sets and epochs can be supplied as well:
~~~
python3 -c 'import nn;nn.test_network(3, sets=5, epochs=500)'
~~~
Input data with x sets can be exported to a file "training_data" with the command
~~~
python3 -c 'import nn;nn.save_training_data(x)'
~~~
Training a new network and exporting it to a file "network_save" can be done with the command, with optional parameters "epochs" and "hidden_layers"
~~~
python3 -c 'import nn;nn.develop_network()'
~~~
To play a game against the saved network, use the command
~~~
python3 nn.py
~~~
Notably, running this command using a network hosting the sigmoid function results in an AI that always places their piece on (0, 0), or the top left hand side of the game board.

## Conclusions and Discussion
After further analysis and reflection, the reason this ANN never learns how to play tic-tac-toe effectively is that resulting values from the output layer are being interpreted incorrectly in the context of tictactoe gameplay. The ANN should be trained such that the output layer represents a probability or some percent value of how a player (or both) is doing in the game. Instead, the code currently treats the output as an estimation of the best next game state, or an estimation of the best move for a player. This output is being interpreted on a scale of 1-9, for each of the nine game tiles.

Moving forward, the ANN has to be modified to "learn" while treating output layer results as a probability instead of as a game state, to be able to have such desired outputs. With those changes, the ANN output could be used in multiple ways as an AI for playing tic-tac-toe. One possible way is to run values through this ANN during a game (after being trained correctly). For each possible move (up to nine), there is a different probability estimate for how well that move will do in the game, or how close that move will be towards winning, or towards losing. Then the best move will be selected for use in the game.

## References
	https://en.wikipedia.org/wiki/Activation_function
	https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

