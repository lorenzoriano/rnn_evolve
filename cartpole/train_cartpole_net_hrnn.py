import sys
sys.path.insert(0, "..")

from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators,  Consts
from pyevolve import DBAdapters
from pyevolve import Crossovers

import numpy as np
import cPickle

import crnn
import pyublas
import crnn.libcartpole

import rnn_evolve
import rnn_evolve.multi_chromosome

test_net = crnn.libcartpole.test_net_single_learning

markov = False
if markov:
    nsensors = 4
else:
    nsensors = 2
maxtrials = 10

cartpole = crnn.libcartpole.SingleCartPole(markov)
cartpole.TAU = 0.05

def chromosome_convert(chromosome):
    input_size = chromosome.getParam("input_size")
    output_size = chromosome.getParam("output_size")
    hidden_size = chromosome.getParam("hidden_size")
    bias_size = hidden_size + output_size

    real_list = chromosome.real_genome.genomeList
    bin_list = chromosome.binary_genome.genomeList    
    signs = np.array([2.*x - 1 for x in bin_list],  dtype = np.float64)
    
    net = crnn.CHRNN(hidden_size,  input_size,  output_size)
    net.learning = False
    net.signs = signs

    array_cr = np.array(real_list[:-bias_size]).reshape( (net.size-net.input_size, net.size) )
    array_bias = np.array(real_list[len(real_list) - bias_size:])
    net.W[net.input_size:, :] = array_cr
    net.bias[net.input_size:] = array_bias
    net.randomiseState()
    return net

def calculate_fitness(net):    
    fitness = test_net(net, cartpole,  maxtrials)
    return fitness

def eval_func(chromosome,  **args):
    net = chromosome_convert(chromosome)
    return calculate_fitness(net)

def callback(ga):
    if not (ga.getCurrentGeneration() % 10):
        sys.stdout.write("Dumping... ")
        best = ga.bestIndividual()
        net = chromosome_convert(best)        
        pickle_file = open("/home/pezzotto/tmp/bestnet.txt", "w")
        cPickle.dump(net,  pickle_file, 2)
        pickle_file.close()        
        sys.stdout.write("done!\t")
        
        fit = calculate_fitness(net)
        sys.stdout.write(" Fitness: %f\n"%( fit, ))
    return False

if __name__ == "__main__":
    # Genome instance
    input_size = nsensors
    hidden_size = 5
    output_size = 1
    bias_size = hidden_size + output_size
    total_size = input_size + hidden_size + output_size
    genome_size = (total_size - input_size)*total_size + bias_size

    genome = rnn_evolve.multi_chromosome.RealBin(genome_size,  total_size)
    genome.real_genome.setParams(rangemin=0, rangemax=1)
    genome.real_genome.setParams(gauss_mu=0., gauss_sigma=0.1)
    genome.setParams(input_size=input_size,  hidden_size=hidden_size,  output_size=output_size)

    genome.real_genome.initializator.set(Initializators.G1DListInitializatorReal)
    genome.real_genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
    genome.real_genome.crossover.set(Crossovers.G1DListCrossoverSinglePoint)
    genome.evaluator.set(eval_func)

    ga = GSimpleGA.GSimpleGA(genome)
#    print "STOPPING ELITISM"
    ga.setElitism(True)
    ga.setElitismReplacement(10)

#    ga.selector.set(Selectors.GTournamentSelectorAlternative)
#    ga.selector.set(Selectors.GRankSelector)
    ga.selector.set(Selectors.GRouletteWheel)
    
    ga.setGenerations(10000)
    ga.setPopulationSize(1000)
    ga.setCrossoverRate(0.1)
    ga.setMutationRate(0.05)
    ga.setMinimax(Consts.minimaxType["maximize"])
    ga.stepCallback.set(callback)
    #sqlite_adapter = DBAdapters.DBSQLite(dbname="/home/pezzotto/Logs/PredatorPrey/evolution.db", identify="pippo")
    #ga.setDBAdapter(sqlite_adapter)

#    ga.setMultiProcessing(True)

    # Do the evolution
    ga.evolve(freq_stats = 10)

    # Best individual
    best = ga.bestIndividual()
    net = chromosome_convert(best)

    pickle_file = open("/home/pezzotto/tmp/bestnet.txt", "w")
    print "Pickling"
    cPickle.dump(net,  pickle_file, 2)


    print "Done"
