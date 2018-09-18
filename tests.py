import unittest
from nn import *

class NNTests(unittest.TestCase):
    def testGeneral(self):
        #neural network for XOR 
        network = NeuralNetwork([2,2,1],0.2) 
        data = []
        for n in [[0,0],[0,1],[1,0],[1,1]]:
            data.append({'inputs':n, 'expected':[n[0]^n[1]]})
        network.train(data, 10000)
        
        for case in data:
            output = round(network.feed_forward(case['inputs'])[0])
            expected = case['expected'][0]
            try:
                self.assertEqual(output, expected, case)
            except:
                save_network(network)
                raise e
    
    def testAgainstNN(self):
        val = train_vs_nn(NeuralNetwork([9,9,1], 0.35), True)

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
