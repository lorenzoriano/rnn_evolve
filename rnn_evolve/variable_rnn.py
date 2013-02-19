# -*- coding: utf-8 -*-
import numpy as np
import scipy.weave as weave
import scipy.weave.converters as converters
import exceptions

DTYPE = np.double

class VariableRNN(object):
    def __init__(self, hidden_size, input_size,  output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.reserved_size = input_size + output_size
        self.size = hidden_size + input_size + output_size
        self.output_slice_left = self.input_size
        self.output_slice_right = self.input_size+self.output_size
        self.hidden_size = hidden_size
        size = self.size
        self.W = np.zeros((size,size), dtype = DTYPE)
        self.bias = np.zeros((size,1), dtype = DTYPE)
        self.x = np.zeros((size, 1), dtype = DTYPE)
#        self.deleted_neurons = []

    def __call__(self,  input):
        if np.any(input > 1.0):
            raise exceptions.ValueError("an input is > 1.0: " + str(input))
        if np.any(input < 0.0):
            raise exceptions.ValueError("an input is < 0.0: " + str(input))

        input = input.ravel()
        if len(input) != self.input_size:
            raise exceptions.ValueError("Input length is " + str(len(input)) + " it should be " + str(self.input_size))

        self.x = self.__c_update(self.x,  self.W,  self.bias,  input,  self.input_size,  self.size)
        return self.x[self.output_slice_left:self.output_slice_right, :]


    def __c_update(self, x,W,bias,input,  input_size,  total_size):
        x_new = np.empty(x.shape,  dtype=DTYPE)
        code = """
        #define sigmoid(x) (1.0/(1+exp(-(x))))
            for (int i = 0; i<input_size; i++)
                x(i,0) = sigmoid(input(i));

            for (int i = input_size; i<total_size; i++) {
                x_new(i,0) = bias(i);
                for (int j = 0; j<total_size; j++) {
                    x_new(i,0) += W(i,j) * x(j,0) ;
                }
                x_new(i,0) = sigmoid(x_new(i,0));
            }
            """
        weave.inline(code, ['x','W','bias','x_new','input', 'input_size', 'total_size'],
                     type_converters=converters.blitz)
        return x_new

    def clone(self):
        newnet = VariableRNN(self.hidden_size,  self.input_size, self.output_size)
        newnet.W = self.W.copy()
        newnet.x = self.x.copy()
        newnet.bias = self.bias.copy()
#        newnet.deleted_neurons = self.deleted_neurons[:]
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
        elif not np.alltrue(self.x == net.x):
            return False
        else:
            return True


    def __str__(self):
        ret = "Size: " + str( self.size)
        ret += " input_size: "  +str(self.input_size)
        ret += " output_size: " +str(self.output_size)
        ret += "\n"
        ret += "W = \n" + str(self.W) + "\nbias = \n" + str(self.bias)
        return ret

    def to_dot(self,  filename):
        try:
            import pygraphviz
        except ImportError:
            print "module pygraphviz not installed"
            return

        graph = pygraphviz.AGraph(strict=False,directed=True)
        nodes_dict = {}
        for i in xrange(self.size):
            node = str(i)
            graph.add_node(node)

        #input
        for i in xrange(self.input_size):
            node = graph.get_node(str(i))
            node.attr["color"] = "red"

        for i in xrange(self.input_size, self.input_size + self.output_size):
            node = graph.get_node(str(i))
            node.attr["color"] = "green"

        for i in xrange(self.size):
            for j in xrange(self.size):
                if self.W[i, j] != 0:
                    label = str("%.2g" % self.W[i, j])
                    graph.add_edge(str(j),  str(i),  label=label)
#                    edge = pydot.Edge(nodes_dict[j],  nodes_dict[i])
#                    edge.set_label = str("%.2g" % self.W[i, j])
#                    graph.add_edge(edge)

        graph.write(filename)

class SelectorRNN(VariableRNN):
    def __init__(self,  hidden_size, left_net,  right_net):
        input_size = left_net.input_size + left_net.output_size*2
        output_size = left_net.output_size
        super(SelectorRNN,  self).__init__(hidden_size, input_size,  output_size)
        self.left_net = left_net
        self.right_net = right_net

    def __call__(self,  input):
        input = input.ravel()
        left_output = self.left_net(input).ravel()
        right_output = self.right_net(input).ravel()

        total_input = np.hstack((left_output,  right_output,  input))
        selectors = super(SelectorRNN, self).__call__(total_input).ravel()
#        print "PEZZOTTO MESSING"
#        selectors -= 0.5
        output = np.empty((1, self.output_size))
        self.weights = []
        for i in xrange(len(selectors)):
            output[0, i] = selectors[i]*left_output[i] + (1.0-selectors[i])*right_output[i]
            self.weights.append(selectors[i])

        return output

    def get_weights(self):
        return self.weights

class SelectorRandomRNN(VariableRNN):
    def __init__(self,  original_net):
        super(SelectorRandomRNN,  self).__init__(original_net.hidden_size, original_net.input_size,  original_net.output_size)

        self.W = original_net.W.copy()
        self.x = original_net.x.copy()
        self.bias = original_net.bias.copy()

        self.left_net = original_net.left_net
        self.right_net = original_net.right_net

    def __call__(self,  input):
        input = input.ravel()
        left_output = self.left_net(input).ravel()
        right_output = np.random.rand(self.right_net.output_size)

        total_input = np.hstack((left_output,  right_output,  input))
#        selectors = super(SelectorRandomRNN, self).__call__(total_input).ravel()
        selectors = [0.5]

        output = np.empty((1, self.output_size))
        self.weights = []
        for i in xrange(len(selectors)):
            output[i] = selectors[i]*left_output[i] + (1.0-selectors[i])*right_output[i]
            self.weights.append(selectors[i])

        return output

    def get_weights(self):
        return self.weights

def generate_random_rnn(hidden_size, input_size,  output_size):
    net = VariableRNN(hidden_size, input_size,  output_size)
    net.W[net.input_size:, :] = np.random.uniform(-1, 1,  (output_size+hidden_size, net.W.shape[1]))
    net.bias[net.input_size:] = np.random.uniform(-1, 1,  (output_size+hidden_size, 1))
    return net

def chromosome_convert(chromosome):
    input_size = chromosome.getParam("input_size")
    output_size = chromosome.getParam("output_size")
    hidden_size = chromosome.getParam("hidden_size")
    bias_size = hidden_size + output_size

    net = variable_rnn.VariableRNN(hidden_size, input_size, output_size)
    array_cr = np.array(chromosome.genomeList[:-bias_size]).reshape( (net.size-net.input_size, net.size) )
    array_bias = np.array(chromosome.genomeList[len(chromosome.genomeList) - bias_size:],  ndmin=2).T
    net.W[net.input_size:, :] = array_cr
    net.bias[net.input_size:] = array_bias
    return net
