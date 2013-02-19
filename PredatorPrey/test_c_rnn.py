from rnn_evolve.c_rnn import c_net_out
from rnn_evolve.c_rnn import c_net_error
#from rnn_evolve.py_rnn import c_net_out
#from rnn_evolve.py_rnn import c_net_error
from rnn_evolve.c_rnn import c_eval_func
import unittest
import numpy as np
import rnn_evolve.variable_rnn as variable_rnn

def chromosome_convert(chromosome,  input_size,  output_size,  hidden_size):
#    input_size = chromosome.getParam("input_size")
#    output_size = chromosome.getParam("output_size")
#    hidden_size = chromosome.getParam("hidden_size")
    bias_size = hidden_size + output_size

    net = variable_rnn.VariableRNN(hidden_size, input_size, output_size)
    array_cr = np.array(chromosome[:-bias_size]).reshape( (net.size-net.input_size, net.size) )
    array_bias = np.array(chromosome[len(chromosome) - bias_size:],  ndmin=2).T
    net.W[net.input_size:, :] = array_cr
    net.bias[net.input_size:] = array_bias
    return net

def eval_func(xs,  ys,  chromosome,  input_size,  output_size,  hidden_size):

    net = chromosome_convert(chromosome,  input_size,  output_size,  hidden_size)
    sum_squared_err = 0
    for x, y in zip(xs, ys):
        net_out = net(x).ravel()
        sum_squared_err += np.linalg.norm(y - net_out)**2

    return sum_squared_err

class TestCCode(unittest.TestCase):
    def setUp(self):
        self.input_size = 200
        self.output_size = 20
        self.hidden_size = 300
        ninputs = 10

        tot_size = self.output_size + self.input_size + self.hidden_size
        list_size = tot_size*(tot_size-self.input_size) + (self.output_size + self.hidden_size)
        self.chromosome = np.random.uniform(-1.0, 1.0, list_size)
        self.input = np.random.uniform(-1.0, 1.0, (ninputs,  self.input_size))
#        self.input = np.array(self.input,  dtype=np.float32)
        self.output = np.random.uniform(-1.0, 1.0, (ninputs, self.output_size))
#        self.output = np.array(self.output,  dtype=np.float32)
        self.net = chromosome_convert(self.chromosome,  self.input_size, self.output_size, self.hidden_size)

    def test_equal_output(self):

        for t in xrange(self.input.shape[0]):
            net_out  = self.net(self.input[t, :]).ravel()
        net_out2 = c_net_out(self.input, self.chromosome,  self.input_size, self.output_size, self.hidden_size)

        self.assertTrue(np.allclose(net_out.ravel(), net_out2.ravel(), 1.e-2, 1e-2))

    def test_equal_error(self):
#        err = eval_func(self.input,  self.output, self.chromosome,  self.input_size, self.output_size, self.hidden_size)
        err = c_eval_func(self.input,  self.output, self.net)
        err1 = c_net_error(self.input,  self.output, self.chromosome,  self.input_size, self.output_size, self.hidden_size)
#        print err1
#        pass

        self.assertTrue(np.allclose(err, err1, 1.e-4, 1e-4))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCCode)
    unittest.TextTestRunner(verbosity=2).run(suite)
#    suite.debug()
