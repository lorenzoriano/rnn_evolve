
from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators,  Consts
from pyevolve import Crossovers

import numpy as np
import cPickle
import math
import sys

import rnn_evolve.variable_rnn as variable_rnn

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

data = np.loadtxt("/home/pezzotto/PythonStuff/train_hand/fake_data.txt", dtype=np.float64)
data = np.vstack((data,data))
input_data = data[:,0]
output_data = data[:,1]

ignore_me = 25
print "Ignoring the first ", ignore_me, " values!!!"

def calculate_fitness(net):

    err = 0.0
    net.x = np.random.rand(net.size,1)
    
    for i in xrange(len(input_data)):
        x = np.array(input_data[i])
        y = np.array(output_data[i])
        netout = net(x)[0]
        if i > ignore_me: #first transient
            err += math.fabs(y - netout)
    
    return err

def eval_func(chromosome,  **args):
    net = chromosome_convert(chromosome)
    return calculate_fitness(net)

def callback(ga):
    if not (ga.getCurrentGeneration() % 10):
        sys.stdout.write("Dumping... ")
        best = ga.bestIndividual()
        net = chromosome_convert(best)
        pickle_file = open("bestnet.txt", "w")
        cPickle.dump(net,  pickle_file, 2)
        pickle_file.close()
        sys.stdout.write("done!\n")
        print "Best chromosome: ", best.genomeList[:]
    return False

if __name__ == '__main__':
    input_size = 1
    hidden_size = 3
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
    genome.crossover.set(Crossovers.G1DListCrossoverTwoPoint)

    ga = GSimpleGA.GSimpleGA(genome)
    ga.setInteractiveMode(False)
    ga.selector.set(Selectors.GRouletteWheel)
    ga.setElitism(True)
    ga.setElitismReplacement(20)
    #ga.selector.set(Selectors.GRankSelector)
    ga.stepCallback.set(callback)
    ga.setGenerations(3000)
    ga.setPopulationSize(200)
    ga.setCrossoverRate(0.2)
    ga.setMutationRate(0.6)
    ga.setMinimax(Consts.minimaxType["minimize"])
#    sqlite_adapter = DBAdapters.DBSQLite(dbname=DIR+"evolution.db", identify="pippo")
#    ga.setDBAdapter(sqlite_adapter)

    # Do the evolution
    ga.evolve(freq_stats = 10)

    # Best individual
    best = ga.bestIndividual()
    print best
    net = chromosome_convert(best)

    pickle_file = open("./bestnet.txt", "w")
    print "Pickling"
    cPickle.dump(net,  pickle_file, 2)
    pickle_file.close()

    print "Done"
