from nn import *
import itertools

def get_genome(net):
    genome = []
    for layer in net.layers:
        for neuron in layer:
            genome.append(neuron.bias)
            for weight in neuron.weights:
                genome.append(weight)
    return genome

def get_net(genome, layer_counts):#for one hidden layer
    iter_count = 0
    neurons = []
    net_layers = [[],[]]
    for hidden in range(layer_counts[1]):
        neuron = Neuron()
        neuron.bias = genome[iter_count]
        iter_count += 1
        weights = []
        for prev_input in range(layer_counts[0]):
            weights.append(genome[iter_count])
            iter_count += 1
        neuron.weights = weights
        net_layers[0].append(neuron)
    for output in range(layer_counts[2]):
        neuron = Neuron()
        neuron.bias = genome[iter_count]
        iter_count += 1
        weights = []
        for prev_input in range(layer_counts[1]):
            weights.append(genome[iter_count])
            iter_count += 1
        neuron.weights = weights
        net_layers[1].append(neuron)
    net = NeuralNetwork(layer_counts)
    net.layers = net_layers
    return net
            
def init_parent(layers):
    net = NeuralNetwork(layers)
    return get_genome(net)

def derive_chromosome(prev_chr, mut_chance, perc_change):
    if random.random() < mut_chance:
        #mutate
        chg = prev_chr * perc_change * (1 if random.random() > 0.5 else -1)
    else:
        chg = 0

    return prev_chr + chg

def make_child(parents, mut_chance, max_perc_chr_chg):
    child = []
    start = 0
    seg_size = len(parents[0])//len(parents)
    for parent in parents:
        for i in range(start, start+seg_size, 1):
            child.append(derive_chromosome(parent[i], mut_chance, max_perc_chr_chg))
    return child

def breed_population(parents, group_size, pop_size, mut_chance, max_perc_chr_chg, parent_percent):
    pop = []
    parent_pair_count = len(parents)//group_size

    for i in range(int(pop_size/parent_pair_count)):
        for j in range(parent_pair_count):
            start = j*group_size
            pop.append(make_child(parents[j:j+group_size], mut_chance, max_perc_chr_chg))
        

    return pop


#play between two neural network players
def play_game(p1, p2):
    board = Board()
    turn_switch = {'o':'x','x':'o'}
    turn = list(turn_switch)[random.randint(0,1)]
    while not board.is_done():
        if turn == 'o':
            board.update(get_network_move(p1, board, 'o'))
        else:
            board.update(get_network_move(p2, board, 'x'))
        turn = turn_switch[turn]

    if board.winner == 'o':
        return 'p1'
    elif board.winner == 'x':
        return 'p2'
    else:
        return 'tie'

def pop_arena(pop, perc_winners, layers):
    print('population size: %s'%(len(pop)))
    winners = []
    count = 0
    scores = [[i] for i in range(len(pop))]
    for pair in itertools.permutations(range(len(pop)), 2):
        p1 = get_net(pop[pair[0]], layers)
        p2 = get_net(pop[pair[1]], layers)
        result = play_game(p1, p2)
        if result == 'p1':
            scores[pair[0]].append(1)
            scores[pair[1]].append(-1)
        elif result == 'p2':
            scores[pair[0]].append(-1)
            scores[pair[1]].append(1)
        else:
            scores[pair[0]].append(0)
            scores[pair[1]].append(0)
        count += 1

    #now find best players
    scores.sort(key=lambda x: sum(x[1:]), reverse=True)
    for i in range(int(len(pop)*perc_winners)):
        winners.append(pop[scores[i][0]])

    print('game count: %s' % (count))
    return {'winners':winners, 'scores':scores}

"""def pop_arena(pop, perc_winners, layers, game_count=100):
    winners = []
    scores = [[i] for i in range(len(pop))]
    for i in range(game_count):
        for i in range(0, len(pop), 2):
            p1 = get_net(pop[i], layers)
            p2 = get_net(pop[i+1], layers)
            result = play_game(p1, p2)
            if result == 'p1':
                scores[i].append(1)
                scores[i+1].append(-1)
            elif result == 'p2':
                scores[i].append(-1)
                scores[i+1].append(1)
            else:
                scores[i].append(0)
                scores[i+1].append(0)
    #now find best players
    scores.sort(key=lambda x: sum(x[1:]), reverse=True)
    for i in range(int(len(pop)*perc_winners)):
        winners.append(pop[scores[i][0]])
    return {'winners':winners, 'scores':scores[:int(len(pop)*perc_winners)]}
"""

def find_player(layers=[9,9,1], max_gen=10, pop_size=50, breed_group_size=2, mut_chance=0.05):
    max_perc_chr_chg = 0.2#chromosomes, if mutating, will change +- up to this percent on a linear scale
    parent_percent = 0.1#what percent of parents move on to reproduce
    parents = [init_parent(layers)]*int(pop_size*parent_percent)

    for gen in range(max_gen):
        population = breed_population(parents, breed_group_size, pop_size, mut_chance, max_perc_chr_chg, parent_percent)
        result = pop_arena(population, parent_percent, layers)
        parents = result['winners']
        print('(%s) top 10 parent scores: %s' % (gen, [sum(x[1:]) for x in result['scores']]))
        

         
    return get_net(parents[0], layers)
