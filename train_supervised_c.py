import sys
sys.path.insert(0, "/home/pezzotto/lib/python2.6/site-packages/PyUblas-0.93.1-py2.6-linux-i686.egg")

from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators,  Consts
from pyevolve import Crossovers

import numpy as np
import cPickle
import math

import crnn
import pyublas

def chromosome_convert(chromosome):
    input_size = chromosome.getParam("input_size")
    output_size = chromosome.getParam("output_size")
    hidden_size = chromosome.getParam("hidden_size")
    bias_size = hidden_size + output_size

    net = crnn.CRNN(hidden_size, input_size, output_size)        
    array_cr = np.array(chromosome.genomeList[:-bias_size]).reshape( (net.size-net.input_size, net.size) )
    array_bias = np.array(chromosome.genomeList[len(chromosome.genomeList) - bias_size:])
    net.W[net.input_size:, :] = array_cr
    net.bias[net.input_size:] = array_bias
    net.x = np.random.rand(*net.x.shape)
    return net

data_in = np.loadtxt("/home/pezzotto/tmp/input.txt", dtype=np.float64)
data_out = np.loadtxt("/home/pezzotto/tmp/output.txt", dtype=np.float64)
data_out = np.array(data_out,  ndmin=2).T

#weights = np.array([1.0 - math.exp(-0.1*x) for x in xrange(len(data_in))],  dtype = np.float64)

def eval_func(chromosome,  **args):
    net = chromosome_convert(chromosome)
#    err = crnn.evaluate_net(net,  data_in,  data_out,  weights)
    err = crnn.evaluate_net(net,  data_in,  data_out)
    return err

def callback(ga):
    if not (ga.getCurrentGeneration() % 10):
        sys.stdout.write("Dumping... ")
        best = ga.bestIndividual()
        net = chromosome_convert(best)
        pickle_file = open("/home/pezzotto/tmp/bestnet.txt", "w")
        cPickle.dump(net,  pickle_file, 2)
        pickle_file.close()
        sys.stdout.write("done!\n")
        print "Best chromosome: ", best.genomeList[:]
    return False

if __name__ == '__main__':
    input_size = data_in.shape[1]
    hidden_size = 4
    output_size = data_out.shape[1]
    
    print "RNN with ",  input_size,  " inputs and ",  output_size, " outpus  and ", hidden_size,  " hiddens"
    
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
    ga.setElitismReplacement(1)
    #ga.selector.set(Selectors.GRankSelector)
    ga.stepCallback.set(callback)
    ga.setGenerations(30000)
    ga.setPopulationSize(600)
    ga.setCrossoverRate(0.0)
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

    pickle_file = open("/home/pezzotto/tmp/bestnet.txt", "w")
    print "Pickling"
    cPickle.dump(net,  pickle_file, 2)
    pickle_file.close()

    print "Done"
