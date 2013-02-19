import numpy as np
import scipy.weave as weave
import scipy.weave.converters as converters
import yapgvb

DTYPE = np.double

class VariableRNN(object):
    def __init__(self, hidden_size, input_size,  output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.reserved_size = input_size + output_size
        self.size = hidden_size + input_size + output_size
        self.hidden_size = hidden_size
        size = self.size
        self.W = np.zeros((size,size), dtype = DTYPE)
        self.bias = np.zeros((size,1), dtype = DTYPE)
        self.x = np.zeros((size, 1), dtype = DTYPE)
        self.deleted_neurons = []

    def evolve(self,  input):
        self.x[:self.input_size, :] = input

#        self.x= self.__c_update(self.x.ravel(),  self.W,  self.bias.ravel(),  input_vector)
        self.x = self.bias + np.dot(self.W,  np.tanh(self.x))
        return self.x

    def __call__(self,  input):
        input = np.array(input,  ndmin = 2,  dtype=DTYPE).T
        x = self.evolve(input)
        return np.tanh(x[self.input_size : self.input_size+self.output_size])

    def clone(self):
        newnet = VariableRNN(self.size - self.input_size - self.output_size,  self.input_size, self.output_size)
        newnet.W = self.W.copy()
        newnet.x = self.x.copy()
        newnet.bias = self.bias.copy()
        newnet.deleted_neurons = self.deleted_neurons[:]
        return newnet

    def __eq__(self,  net):
        if not self.reserved_size == net.reserved_size:
            return False
        elif not self.size == net.size:
            return False
        elif not self.input_size == net.input_size:
            return False
        elif not self.output_size == net.output_size:
            return False
        elif not np.alltrue(self.W == net.W):
            return False
        elif not np.alltrue(self.bias == net.bias):
            return False
        elif not self.deleted_neurons == net.deleted_neurons:
            return False
        elif not np.alltrue(self.x == net.x):
            return False
        else:
            return True

    def __c_update(self, x,W,b,  input):
        x_new = np.empty(x.shape[0],  dtype=DTYPE)

        code = """
            #define f(x) tanh(x)
            #define prod(W,_x) sum(W(i,j) * _x(j),j)

            using namespace blitz;
            firstIndex i;
            secondIndex j;
            x = f(x);
            x_new = b + input + prod(W, x);
            """
        weave.inline(code, ['x','W','b','x_new','input'],
                     type_converters=converters.blitz)
        return x_new.ravel()

    def __str__(self):
        ret = "Size: " + str( self.size)
        ret += " input_size: "  +str(self.input_size)
        ret += " output_size: " +str(self.output_size)
        ret += "\n"
        ret += "W = \n" + str(self.W) + "\nbias = \n" + str(self.bias)
        return ret

    def to_dot(self,  filename):
        graph = yapgvb.Digraph("RNN")
        nodes_dict = {}
        for i in xrange(self.size):
            nodes_dict[i] = graph.add_node(str(i))

        #input
        for i in xrange(self.input_size):
            nodes_dict[i].color="red"

        for i in xrange(self.input_size, self.input_size + self.output_size):
            nodes_dict[i].color="green"

        for i in xrange(self.size):
            for j in xrange(self.size):
                if self.W[i, j] != 0:
                    edge = graph.add_edge(nodes_dict[j],  nodes_dict[i])
                    edge.label = str("%.2g" % self.W[i, j])

        graph.layout(yapgvb.engines.dot)
        graph.render(filename, format="dot")

def generate_random_rnn(hidden_size, input_size,  output_size):
    net = VariableRNN(hidden_size, input_size,  output_size)
    net.W[input_size:, :] = np.random.uniform(-1, 1,  (output_size+hidden_size, net.W.shape[1]))
    net.bias[input_size:] = np.random.uniform(-1, 1,  (output_size+hidden_size, 1))
    return net
