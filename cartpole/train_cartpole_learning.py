import sys
sys.path.insert(0, "..")

from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators,  Consts
from pyevolve import DBAdapters
from pyevolve import Crossovers
import rnn_evolve
import rnn_evolve.multi_chromosome

import numpy as np
import cPickle

import crnn
import pyublas
import crnn.libcartpole

test_net = crnn.libcartpole.test_net_single_learning

markov = False
if markov:
    nsensors = 4
else:
    nsensors = 2
maxtrials = 50

def calculate_fitness(net):    
    fitness = test_net(net, maxtrials,  markov)
    return fitness

def chromosome_convert(chromosome):
    input_size = chromosome.getParam("input_size")
    output_size = chromosome.getParam("output_size")
    hidden_size = chromosome.getParam("hidden_size")
    
    real_list = chromosome.real_genome.genomeList
    bin_list = chromosome.binary_genome.genomeList
    
    signs = [2*x - 1 for x in bin_list]
    
    net = crnn.CHRNN.build_from_list(real_list, signs, hidden_size,  input_size,  output_size)
    return net

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
        print "Real Genome: ",  best.real_genome.genomeList
        print "Binary Genome: ",  best.binary_genome.genomeList
        
    return False

if __name__ == "__main__":
    # Genome instance
    input_size = nsensors
    hidden_size = 3
    output_size = 1
    
    bias_size = hidden_size + output_size
    total_size = input_size + hidden_size + output_size

    genome = rnn_evolve.multi_chromosome.RealBin(total_size,  total_size)
    genome.real_genome.setParams(rangemin=0, rangemax=1)
    genome.real_genome.setParams(gauss_mu=0., gauss_sigma=0.1)
    genome.setParams(input_size=input_size,  hidden_size=hidden_size,  output_size=output_size)

    genome.real_genome.initializator.set(Initializators.G1DListInitializatorReal)
    genome.real_genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
    genome.real_genome.crossover.set(Crossovers.G1DListCrossoverSinglePoint)
    genome.evaluator.set(eval_func)

    ga = GSimpleGA.GSimpleGA(genome)
    print "STOPPING ELITISM"
    ga.setElitism(False)
#    ga.setElitismReplacement(1)

#    ga.selector.set(Selectors.GTournamentSelectorAlternative)
#    ga.selector.set(Selectors.GRankSelector)
    ga.selector.set(Selectors.GRouletteWheel)
    
    ga.setGenerations(10000)
    ga.setPopulationSize(500)
    ga.setCrossoverRate(0.2)
    ga.setMutationRate(0.1)
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
