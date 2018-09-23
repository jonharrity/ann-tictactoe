import unittest
from nn import *
from genetic import *

random.seed(1002)

class GeneticTests(unittest.TestCase):
    def testGenomeConverter(self):
        genome = get_genome(NeuralNetwork([9,9,1]))
        self.assertEqual(len(genome), 100)
        for layers in [[1,1,1], [9,8,1], [2,4,6]]:
            net_start = NeuralNetwork(layers)
            genome = get_genome(net_start)
            net_end = get_net(genome, layers)
            self.assertEqual(net_start.to_string(), net_end.to_string())

    def testInitParent(self):
        for layers in [[1,1,1], [9,9,1]]:
            genome = init_parent(layers)
            self.assertEqual(get_net(genome, layers).get_layer_counts(), layers)


    def testFindPlayer(self):#genetic training entrance function
        net = find_player() 
        self.assertEqual(type(net), NeuralNetwork)
        save_network(net)
        print('saved network: %s'%(net))

    def testMakeChild(self):
        layers = [9,5,1]
        parent_count = 2
        mut_chance = 0.2
        chr_change = 0.2
        for i in range(10):
            parents = [init_parent(layers) for i in range(parent_count)]
            child = make_child(parents, mut_chance,chr_change)
            self.assertEqual(len(child), len(parents[0]))

    def testDeriveChromosome(self):
        mut_chance = 0.2
        chr_chg = 0.2
        chg_allowance = 0.01
        for i in range(100):
            prev = random.randint(-10,100)
            new = derive_chromosome(prev, mut_chance, chr_chg)
            diff = new - prev
            if diff != 0:
                self.assertLessEqual(abs(diff/prev), chr_chg+chg_allowance)

    def testArena(self):
        parent_count = 10
        layers = [9,5,1]
        parents = [init_parent(layers)]*parent_count
        group_size = 2#2 parents per child
        pop_size=100
        mut_chance=0.2
        max_perc_chr_chg=0.2
        parent_percent=0.1
        pop = breed_population(parents,group_size, pop_size, mut_chance,max_perc_chr_chg,parent_percent)
        
        result = pop_arena(pop, parent_percent, layers)
        new_parents = result['winners']
        scores = result['scores']
        self.assertEqual(len(new_parents), parent_count)
        prev_total = sum(scores[0][1:])
        for score in scores:
            new_score = sum(score[1:])
            self.assertLessEqual(new_score, prev_total)
            prev_total = new_score


    def testBreedPopulation(self):
        layers = [9,5,1]
        parents = [init_parent(layers)]*10
        group_size = 2#2 parents per child
        pop_size=100
        mut_chance=0.2
        max_perc_chr_chg=0.2
        parent_percent=0.1
        pop = breed_population(parents,group_size, pop_size, mut_chance,max_perc_chr_chg,parent_percent)
        self.assertEqual(len(pop), pop_size)


class NNTests(unittest.TestCase):
    def testXOR(self):
        #neural network for XOR 
        network = NeuralNetwork([2,2,1],0.2) 
        data = []
        for n in [[0,0],[0,1],[1,0],[1,1]]:
            data.append({'inputs':n, 'expected':[n[0]^n[1]]})
        network.train(data, 10000)
        
        for case in data:
            output = round(network.feed_forward(case['inputs'])[0])
            expected = case['expected'][0]
            self.assertEqual(output, expected, case)
    
    def testAgainstNN(self):
        return
        val = train_vs_nn(NeuralNetwork([9,10,1], 0.35), True)

    def testRotateBoardClockwise(self):
        board = Board()
        tiles = ['o','x']
        pers = tiles[random.randint(0,1)]
        for i in range(5):#fill 5 random tiles
            opts = list(board.get_empty_tiles())
            tile = opts[random.randint(0,len(opts)-1)]
            board.update(tile[0],tile[1],tiles[random.randint(0,1)])
        prev = board.export_for_nn(pers)
        for i in range(3):  
            rotated = export_clockwise_rotation(prev)
            self.assertNotEqual(prev, rotated)
            prev = rotated
        self.assertEqual(export_clockwise_rotation(prev), board.export_for_nn(pers)) 



if __name__ == '__main__':
    unittest.main()
