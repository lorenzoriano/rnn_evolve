import numpy as np

def delete_synapses(net,  prob):
    ind = np.random.uniform(size = net.W.shape) <= prob
    net.W[ind] = 0.0

def insert_synapses(net,  prob):
    #insert values where W=0
    is_absent = net.W == 0.0
    ind = np.random.uniform(size = net.W.shape) <= prob
    mix = np.logical_and(is_absent,  ind)
    mix[:net.input_size, :] = False
    mix[net.deleted_neurons, :] = False
    mix[:, net.deleted_neurons] = False
    net.W[mix] = np.random.uniform(-1, 1, size = net.W.shape)[mix]*prob

def modify_synapses(net, prob,  sigma_w):
    #change weight of non-zero synapses
    is_present = net.W != 0.0
    ind = np.random.uniform(size = net.W.shape) <= prob
    mix = np.logical_and(is_present,  ind)
    mix[:net.input_size, :] = False
    mix[net.deleted_neurons, :] = False
    mix[:, net.deleted_neurons] = False
    net.W[mix] += np.random.normal(0, sigma_w, size = net.W.shape)[mix]*prob

def modify_neurons(net,  prob,  sigma_b):
    ind = np.random.uniform(size = len(net.x)) <= prob
    ind[:net.input_size] = False
    delta_b = np.random.normal(0, sigma_b, net.bias.shape)
    net.bias[ind] += delta_b[ind]
    net.bias[net.deleted_neurons] = 0

def delete_neurons(net, prob):
    ind = np.random.uniform(size = len(net.x)) <= prob
    ind[:net.reserved_size] = False
    ind[net.deleted_neurons] = False
    net.W[ind, :] = 0.0
    net.W[:,  ind] = 0.0
    net.bias[ind] = 0.0

    non_zero_list = [i for i in np.nonzero(ind)[0]]
    net.deleted_neurons += non_zero_list
    net.size -= len(non_zero_list)

def insert_neuron(net,  prob):
    if np.random.uniform() <= prob:
        newbias = np.random.uniform(-1, 1)
        nneurons = len(net.W)

        #take an empty slot
        if len(net.deleted_neurons):
            to_add = net.deleted_neurons.pop()
            newcol = np.random.uniform(-1, 1,   nneurons)
            newcol[:net.input_size] = 0
            newcol[net.deleted_neurons] = 0
            newrow = np.random.uniform(-1, 1,  nneurons )
            newrow[net.deleted_neurons] = 0
            net.W[to_add, :] = newrow
            net.W[:, to_add] = newcol
            net.bias[to_add] = newbias

        #add a new row/column
        else:
            newcol = np.random.uniform(-1, 1,  (nneurons, 1))
            newcol[:net.input_size, :] = 0
            newrow = np.random.uniform(-1, 1,  (1, nneurons+1) )
            net.W = np.hstack( (net.W,  newcol) )
            net.W = np.vstack( (net.W,  newrow) )

            net.bias = np.vstack( (net.bias, newbias))
            net.x = np.vstack( (net.x, 0) )

        net.size += 1

def mutate(individual,  **args):
    if not individual.mutable:
        return
    net = individual.RNN
    neurons_delete_prob = args.get("neurons_delete_prob")
    neuron_insert_prob = args.get("neuron_insert_prob")
    neurons_modify_prob = args.get("neurons_modify_prob")
    synapses_delete_prob = args.get("synapses_delete_prob")
    synapses_insert_prob = args.get("synapses_insert_prob")
    synapses_modify_prob = args.get("synapses_modify_prob")
    neurons_sigma = args.get("neurons_sigma")
    synapses_sigma = args.get("synapses_sigma")

    delete_synapses(net, synapses_delete_prob)
    insert_synapses(net, synapses_insert_prob)
    modify_synapses(net, synapses_modify_prob,  synapses_sigma)

    delete_neurons(net, neurons_delete_prob)
    insert_neuron(net, neuron_insert_prob)
    modify_neurons(net, neurons_modify_prob, neurons_sigma)
