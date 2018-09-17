from tictactoe import * 
from math import *
import random
import pickle

random.seed(8080)


class Neuron:
    def to_string(self):
        return 'neuron with weights %s and bias %s' % (self.weights, self.bias)
    def __repr__(self):
        return self.to_string()
    def __init__(self, weights=None):
        if weights:
            self.weights = weights


class NeuralNetwork:
    def to_string(self):
        parts = []
        for layer in self.layers:
            parts.append('[%s]' % ', '.join(neuron.to_string() for neuron in layer))
        return 'Neural network with %s layers: %s' % (len(self.layers), '[%s]' % ', '.join(parts))

    #only call after feed_forward, get array output of ANN
    def get_output(self):
        return [n.output for n in self.layers[-1]]
    def __str__(self):
        return 'neural network with layers %s' % str([len(self.layers[0][0].weights)]+[len(n) for n in self.layers])
    def __repr__(self):
        return self.__str__()
        

    def feed_forward(self, inputs):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            next_inputs = []
            for neuron in layer:
                #self.biases[i] (layer bias) > neuron.bias (neuron bias) > 0 (disable)
                activation = -neuron.bias
                for x in range(len(inputs)):
                    if x >= len(neuron.weights):
                        raise Exception('invalid input: %s inputs supplied, but only %s nodes in first layer' % (range(len(inputs)), len(neuron.weights)))
                    activation += inputs[x] * neuron.weights[x]
                neuron.output = self.transfer(activation)
                next_inputs.append(neuron.output)

            inputs = next_inputs
        return self.get_output()
    
    def back_propogate(self, expected, inputs):
        error_sum = 0
        
        for i in range(len(expected)):
            actual = self.layers[-1][i].output
            error = expected[i] - actual
            error_sum += pow(error, 2)
            output_gradient = self.transfer_derivative(actual) * error
            #loop through weights between hidden layer
            for j in range(len(self.layers[-2])):
                previous_output = self.layers[-2][j].output
                delta = self.learning_rate * previous_output * output_gradient
                self.layers[-1][i].weights[j] += delta
                hidden_gradient = self.transfer_derivative(previous_output) * output_gradient * previous_output
                #loop through weights between input layer
                for k in range(len(self.layers[0][0].weights)):
                    delta = self.learning_rate * inputs[k] * hidden_gradient
                    self.layers[-2][j].weights[k] += delta

                #update bias for hidden node
                delta = self.learning_rate * hidden_gradient
                self.layers[-2][j].bias -= delta
            
            #update bias for output node
            delta = self.learning_rate * output_gradient
            self.layers[-1][i].bias -= delta

        return error_sum


    """
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
   
    def transfer(self, x):
        #print('transfer %s'%x)
        return tanh(x)
    def transfer_derivative(self, x):
        return 1-tanh(x)**2
    """
    def transfer(self, x):
        try:
            return 1 / (1+exp(-x))
        except:
            print('math overflow for transfer(%s)'%x)
            if x > 10: return 1
            else: return 0

    def transfer_derivative(self, x):
        #return x * (1 - x)
        return self.transfer(x) * (self.transfer(-1*x))

    def __init__(self, layer_counts, learning_rate=0.35):#initialize network with random weights and biases
        self.layers = []
        self.biases = []
        input_count = layer_counts[0]
        layer_counts = layer_counts[1:]
        for size in layer_counts:
            new_layer = []
            for i in range(size):
                neuron = Neuron()
                neuron.weights = [random.random() for i in range(input_count)]
                neuron.bias = random.random()
                new_layer.append(neuron)
            self.layers.append(new_layer)
            self.biases.append(0.5)
            input_count = size
        self.learning_rate = learning_rate

    def update_weights(self, inputs):
        layer_count = 1
        for layer in self.layers:
            next_inputs = []
            sav_changes = []
            count_neurons_changed = 0
            count_neurons_unchanged = 0
            for neuron in layer:
                total_change = 0
                for i in range(len(inputs)):
                    weight_start = neuron.weights[i]
                    neuron.weights[i] -= inputs[i] * neuron.delta * self.learning_rate * neuron.output
                    weight_end = neuron.weights[i]
                    total_change += abs(weight_start - weight_end)
                    if weight_start == weight_end:
                        count_neurons_unchanged += 1
 #                       print('unchanged neuron, weight: %s' % weight_start)
                    else:
                        count_neurons_changed += 1
                        #print('change neuron, delta %s, output %s, input %s %s'%(neuron.delta, neuron.output, inputs[i], self.biases[layer_count-1]))

                #self.biases[layer_count-1] (layer bias) > neuron.bias
                self.biases[layer_count-1] -= neuron.delta * self.learning_rate
                sav_changes.append(total_change)
                
                next_inputs.append(neuron.output)
#            print('layer %s ; weight changes (%s / %s) %s' % (layer_count, count_neurons_changed, (count_neurons_changed + count_neurons_unchanged), sav_changes))
            inputs = next_inputs
            layer_count += 1

    def train(self, dataset, epochs):
        print('calling train, dataset len %s, epochs: %s' % (len(dataset), epochs))
        if type(dataset) != type([]):
            raise Exception('NeuralNetwork.train: dataset not an array')
        if len(dataset) < 1:
            raise Exception('NeuralNetwork.train: empty dataset provided')
        if type(dataset[0]['expected']) != type([]):
            raise Exception('NeuralNetwork.train: \'expected\' value is not an array')

        print('training, learning rate = %s' % self.learning_rate)

        for gen in range(epochs):
            dataset.sort(key=lambda x:random.random())
            #print('training dataset %s'%dataset)
            for training_set in dataset:
                error = 0
                self.feed_forward(training_set['inputs'])
                error = self.back_propogate(training_set['expected'], training_set['inputs'])
            if gen == 0:
                print('first epoch: error = %s' % str(error))
            if gen+1 == epochs:
                print('last epoch: error = %s' % str(error))


def test_network(hidden_layers, sets=10, epochs=100):
    data = get_training_data(sets)
    network = NeuralNetwork([18,hidden_layers,1], 0.5)
    network.train(data, epochs)


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

def get_all_training_data(board=None, current_player='x'):
    data = []
    turn_switch = {'x':'o','o':'x'}

    if board == None:
        print('creating training data from all possible game states')
        board = Board()
    else:
        if board.is_done():
            return data


    for tile in board.get_empty_tiles():
        new_board = board.copy()
        new_board.update((tile[0],tile[1],current_player))
        data.append({'inputs':new_board.export_for_nn(current_player),'expected':new_board.get_health(current_player)})


        data.extend(get_training_data(new_board, turn_switch[current_player]))
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
    options = []
    for tile in board.get_empty_tiles():
        test_board = board.copy()
        test_board.update((tile[0],tile[1],pov))
        network.feed_forward(test_board.export_for_nn(pov))
        options.append({'x':tile[0],'y':tile[1],'val': network.layers[-1][0].output})
    options.sort(key=lambda x: x['val'])
    print('got options: %s'%options)
    return (options[-1]['x'], options[-1]['y'], pov)

def human_vs_nn(network=load_network()):
    board = Board()
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


def get_xor_data():
    return [{'inputs':n[:2],'expected':[n[2]]} for n in [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]]


if __name__ == '__main__':
    human_vs_nn()



def get_training_data(sets):
    print('creating training data from  %s games'%sets)
    data = []
    for i in range(sets):
        board = Board()
        turn_switch = {'o': 'x', 'x': 'o'}
        turn = ['o', 'x'][random.randint(0,1)]
        ai_letter = ['o', 'x'][random.randint(0,1)]
        nn_pov = ['o', 'x'][random.randint(0,1)]
        done = False
        while not done:
            #only add data if its AI turn
            if turn == nn_pov:
                data.append({'inputs':board.export_for_nn(ai_letter),
                            'expected': [board.get_health(ai_letter)]})

            if turn == ai_letter: # minimax turn
                move = get_max(board, None, 1, -1000, 1000, ai_letter)[0]
                board.update((move[0], move[1], ai_letter))
            else:#random turn
                board.update(get_random_move(board, turn))
            turn = turn_switch[turn]
            done = board.is_done()
    return data



