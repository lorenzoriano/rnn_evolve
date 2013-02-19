from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators,  Consts
from pyevolve import DBAdapters
from pyevolve import Crossovers

import numpy as np
import pickle
import random

import rnn_evolve.variable_rnn as variable_rnn
from rnn_evolve.c_rnn import c_eval_func
from rnn_evolve.kolmogorov import kolmogorov_zlib

data = np.loadtxt("/home/pezzotto/tmp/all_inputs.txt")

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

def eval_func(chromosome,  **args):
    nsamples = data.shape[0]
    net = chromosome_convert(chromosome)
    net_out = np.empty(nsamples, dtype=np.int)

    for i in xrange(nsamples):
        x = data[i, :]
        out = net(x).ravel()
        net_out[i] = out.ravel()*100

    return kolmogorov_zlib(net_out.ravel())


# Genome instance
input_size = data.shape[1]
hidden_size = 2
output_size = 1
bias_size = hidden_size + output_size
total_size = input_size + hidden_size + output_size
genome_size = (total_size - input_size)*total_size + bias_size

genome = G1DList.G1DList(genome_size)
genome.setParams(rangemin=-3, rangemax=3)
genome.setParams(input_size=input_size,  hidden_size=hidden_size,  output_size=output_size)
genome.setParams(gauss_mu=0., gauss_sigma=1.)

genome.initializator.set(Initializators.G1DListInitializatorReal)
genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
genome.evaluator.set(eval_func)
genome.crossover.set(Crossovers.G1DListCrossoverSinglePoint)

ga = GSimpleGA.GSimpleGA(genome)
ga.selector.set(Selectors.GRouletteWheel)
#ga.selector.set(Selectors.GRankSelector)
ga.setGenerations(1500)
ga.setPopulationSize(300)
ga.setCrossoverRate(0.2)
ga.setMutationRate(0.8)
ga.setMinimax(Consts.minimaxType["maximize"])
#sqlite_adapter = DBAdapters.DBSQLite(dbname="/home/pezzotto/Logs/RobotFollowing/evolution.db", identify="hidden-10")
#ga.setDBAdapter(sqlite_adapter)

# Do the evolution
ga.evolve(freq_stats = 10)

# Best individual
best = ga.bestIndividual()
net = chromosome_convert(best)
#net.to_dot("/home/pezzotto/Logs/RobotFollowing/net.dot")

pickle_file = open("/home/pezzotto/Logs/PredatorPrey/complex_net.txt", "w")
print "Pickling"
pickle.dump(net,  pickle_file, 2)

#init = 0.5
#values = np.empty(100)
#values[0] = init
#for t in xrange(1, len(values)):
#    values[t] = net(values[t-1])
#
#np.savetxt("/home/pezzotto/data.txt",  values)

#pickle_file = open("/home/pezzotto/Logs/RobotFollowing/try.txt", "r")
#net = pickle.load(pickle_file)
#pickle_file.close()
#print net


print "Done"
