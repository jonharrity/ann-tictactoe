from tictactoe import * 
from math import *
import random
import pickle

pov_def = {'x': 0.5, 'o': 0.75}

class Neuron:
    def to_string(self):
        return 'weights %s, bias %s' % (self.weights, self.bias)


class NeuralNetwork:
    def to_string(self):
        parts = []
        for layer in self.layers:
            parts.append('[%s]' % ', '.join(neuron.to_string() for neuron in layer))
        return 'Neural network with %s layers: %s' % (len(self.layers), '[%s]' % ', '.join(parts))

    def feed_forward(self, inputs):
        for layer in self.layers:
            next_inputs = []
            for neuron in layer:
                activation = neuron.bias
                for x in range(len(inputs)):
                    activation += inputs[x] * neuron.weights[x]
                neuron.output = self.transfer(activation)
                next_inputs.append(neuron.output)

            inputs = next_inputs

    def back_propogate(self, expected):
        for i in range(len(expected)):#first find error of output layer
            neuron = self.layers[-1][i]
            neuron.delta = (expected[i] - neuron.output) * self.transfer_derivative(neuron.output)

        for ilayer in range(len(self.layers)-2,-1,-1):#find error of hidden layers
            for ineuron in range(len(self.layers[ilayer])):
                neuron = self.layers[ilayer][ineuron]
                error = 0
                for next_neuron in self.layers[ilayer+1]:
                    error += next_neuron.delta * next_neuron.weights[ineuron]
                neuron.delta = error * self.transfer_derivative(neuron.output)
   
    """
    def transfer(self, x):
        return tanh(x)
    def transfer_derivative(self, x):
        return 1-tanh(x)**2
    """
    def transfer(self, x):
        return 1 / (1+exp(x))
    def transfer_derivative(self, x):
        return self.transfer(x) * (1-self.transfer(x))
    
    def __init__(self, layer_counts, learning_rate):#initialize network with random weights and biases
        layers = []
        input_count = layer_counts[0]
        layer_counts = layer_counts[1:]
        for size in layer_counts:
            new_layer = []
            for i in range(size):
                neuron = Neuron()
                neuron.bias = 1
                neuron.weights = [random.random() for i in range(input_count)]
                new_layer.append(neuron)
            layers.append(new_layer)
            input_count = size
        self.layers = layers
        self.learning_rate = learning_rate

    def update_weights(self, inputs):
        for layer in self.layers:
            next_inputs = []
            for neuron in layer:
                for i in range(len(inputs)):
                    neuron.weights[i] += inputs[i] * neuron.delta * self.learning_rate * neuron.output
                neuron.bias += neuron.delta * self.learning_rate
                next_inputs.append(neuron.output)
            inputs = next_inputs

    def train(self, dataset, epochs):
        print('training, learning rate = %s' % self.learning_rate)
        for gen in range(epochs):
            error = 0
            for training_set in dataset:
                self.feed_forward(training_set['inputs'])
                self.back_propogate(training_set['expected'])
                self.update_weights(training_set['inputs'])
                error += sum((self.layers[-1][i].delta)**2 for i in range(len(training_set['expected'])))
                print('output %s for expected %s' % (self.layers[-1][0].output, training_set['expected']))
            print('epoch %s: error is %s' % (str(gen), error))


def test_network(hidden_layers, sets=10, epochs=100):
    data = get_training_data(sets)
    network = NeuralNetwork([10,hidden_layers,1], 0.5)
    print(network.to_string())
    network.train(data, epochs)
    print(network.to_string())


#INPUTS DEFINITION:
#EMPTY: 0
#X: 1
#O: 2
def get_inputs_board(board):
    input_switch = {EMPTY: 0.25, 'x': 0.5, 'o': 0.75}
    inputs = []
    for y in range(3):
        for x in range(3):
            inputs.append(input_switch[board.tiles[x][y]])
    return inputs

def get_best_move(board, team):
    move = get_max(board, None, 1, -1000, 1000, team)[0]# move is (x, y)
    val = move[0] + move[1]*3
    return val / 8

def get_random_move(board, turn):
    options = [n for n in board.get_empty_tiles()]
    move = list(options[random.randint(0,len(options)-1)]) + [turn]
    return move

def permute(data_set, first):
    if len(data_set) == 1:
        return first + data_set[0]
    for i in range(len(data_set)):
        for z in permute(first[0:i]+first[i+1:], first+n):
            yield z

#generate training data for "sets" number of games, with one player being random, one using minimax
def get_training_data(sets):
    print('creating %s sets of training data'%sets)
    data = []
    for i in range(sets):
        board = Board()
        turn_switch = {'o': 'x', 'x': 'o'}
        turn = ['o', 'x'][random.randint(0,1)]
        ai_letter = ['o', 'x'][random.randint(0,1)]
        nn_pov = ['o', 'x'][random.randint(0,1)]
        done = False
        while not done:
            if turn == nn_pov:
                data.append({'inputs':get_inputs_board(board)+[pov_def[nn_pov]], 'expected': [get_best_move(board, turn)]})
            
            if turn == ai_letter: # minimax turn
                move = get_max(board, None, 1, -1000, 1000, ai_letter)[0]
                board.update((move[0], move[1], ai_letter))
            else:#random turn
                board.update(get_random_move(board, turn))
            turn = turn_switch[turn]
            done = board.is_done()
     
    return data

training_data_file = 'training_data'
nn_file = 'network_save'

def save_random_data(sets):
    data = []
    for n in range(sets):
        inputs = [random.randint(0,2) for i in range(9)] + [random.randint(0,1)]
        expected = [random.randint(0,8)]
        data.append({'inputs': inputs, 'expected':expected})
    file = open(training_data_file,'wb')
    pickle.dump(data,file)
    file.close()

def load_training_data():
    file = open(training_data_file, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def save_training_data(sets):
    data = get_training_data(sets)
    file = open(training_data_file, 'wb')
    pickle.dump(data,file)
    file.close()

def save_network(network):
    file = open(nn_file, 'wb')
    pickle.dump(network, file)
    file.close()

def load_network():
    file = open(nn_file, 'rb')
    network = pickle.load(file)
    file.close()
    return network


def get_network_move(network, board, pov):
    network.feed_forward(get_inputs_board(board) + [pov_def[pov]])
    output = network.layers[-1][0].output
    print('rouding output %s to %s' % (output, round(output)))
    output = round(output)
    x = output % 3
    y = output // 3
    return (x, y, pov)

def human_vs_nn():
    board = Board()
    network = load_network()
    turn_switch = {'x':'o', 'o':'x'}
    turn = ['o','x'][random.randint(0,1)]
    human_piece = ['o','x'][random.randint(0,1)]
    done = False
    while not done:
        if turn == human_piece:
            print('Your turn. Current board:')
            board.print()
            move = get_human_move(board)
            board.update((move[0],move[1],turn))
        else:
            print('Neural network move; board:')
            board.print()
            move = get_network_move(network, board, turn)
            print('got nn move %s' % str(move))
            board.update(move)
            print('Computer moved to (%s, %s)'%(str(move[0]),str(move[1])))
        done = board.is_done()
        turn = turn_switch[turn]
        print()
    board.print()
    print()
    board.print_win_msg(human_piece, turn_switch[human_piece])
    

def develop_network(epochs=500, hidden_layers=5): 
    layer_counts = [10, hidden_layers, 1]
    learning_rate = 0.5

    nn = NeuralNetwork(layer_counts, learning_rate)
    training_data = load_training_data() 
    nn.train(training_data, epochs)
    save_network(nn)



if __name__ == '__main__':
    human_vs_nn()



