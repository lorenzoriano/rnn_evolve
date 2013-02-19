import numpy as np
import math

def c_net_out(xs,  chromosome,  input_size,  output_size,  hidden_size):

        xs = np.array(xs)
#        input_size = chromosome.getParam("input_size")
#        output_size = chromosome.getParam("output_size")
#        hidden_size = chromosome.getParam("hidden_size")
        bias_size = hidden_size + output_size
        total_size = bias_size + input_size

#        chromosome_list = chromosome.genomeList
        chromosome_list = chromosome
        bias_starting_point = len(chromosome_list) - bias_size

        state = np.zeros(total_size,  dtype = np.float32)
        newstate = np.zeros(total_size,  dtype = np.float32)

        for t in xrange(xs.shape[0]):
            x = xs[t, :]

            newstate = state.copy()
            for i in xrange(input_size):
                state[i] = x[i]
                newstate[i] = x[i]

            for i in xrange(input_size,  total_size):
                #bias
                newstate[i] = chromosome_list[len(chromosome) - bias_size + i - input_size]
                for j in xrange(0,  total_size):
                    #x[i] += w[i,j]*tanh(x[j])
                    newstate[i] += chromosome_list[ (i-input_size) * (total_size) + j] * math.tanh(state[j])

            state = newstate.copy()

        out = np.empty(output_size,  dtype = np.float32)
        for i, j in zip(xrange(input_size, input_size + output_size), xrange(len(out)) ):
            out[j] = math.tanh(state[i])

        return out


def c_net_error(xs,  ys,  chromosome,  input_size,  output_size,  hidden_size):

        xs = np.array(xs)
        ys = np.array(ys)
#        input_size = chromosome.getParam("input_size")
#        output_size = chromosome.getParam("output_size")
#        hidden_size = chromosome.getParam("hidden_size")
        bias_size = hidden_size + output_size
        total_size = bias_size + input_size

#        chromosome_list = chromosome.genomeList
        chromosome_list = chromosome
        bias_starting_point = len(chromosome_list) - bias_size

        state = np.zeros(total_size,  dtype = np.float32)
        newstate = np.zeros(total_size,  dtype = np.float32)
        out = np.empty(output_size,  dtype = np.float32)
        sse = 0

        for t in xrange(xs.shape[0]):
            x = xs[t, :]
            y = ys[t, :]

            newstate = state.copy()
            for i in xrange(input_size):
                state[i] = x[i]
                newstate[i] = x[i]

            for i in xrange(input_size,  total_size):
                newstate[i] = chromosome_list[len(chromosome) - bias_size + i - input_size]
                for j in xrange(0,  total_size):
                    #x[i] += w[i,j]*tanh(x[j])
                    newstate[i] += chromosome_list[ (i-input_size) * (total_size) + j] * math.tanh(state[j])

            state = newstate.copy()
            for i, j in zip(xrange(input_size, input_size + output_size), xrange(len(out)) ):
                out[j] = math.tanh(state[i])
            error_t = 0
            for i in xrange(output_size):
                error_t += (out[i] - y[i])*(out[i] - y[i])
            sse += error_t

        return sse

