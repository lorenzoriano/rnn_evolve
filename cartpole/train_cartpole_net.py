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

print "Evaluating with centering fitness"
test_net = crnn.libcartpole.evaluate_cartpole_center

#print "Evaluating with normal fitness"
#test_net = crnn.libcartpole.evaluate_cartpole_single

print "Cartpole is non Markov"
markov = False
if markov:
    nsensors = 4
else:
    nsensors = 2
maxtrials = 10

cartpole = crnn.libcartpole.SingleCartPole(markov)
cartpole.TAU = 0.02
cartpole.FORCE_MAG = 100.0
print "Cartpole TAU is ",  cartpole.TAU
print "Cartpole FORCE_MAG is ",  cartpole.FORCE_MAG


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
        
        fit = crnn.libcartpole.evaluate_cartpole_single(net,  cartpole,  maxtrials)
        sys.stdout.write(" Steps: %f\n"%( fit, ))
    return False

if __name__ == "__main__":
    # Genome instance
    input_size = nsensors
    hidden_size = 4
    output_size = 1
    bias_size = hidden_size + output_size
    total_size = input_size + hidden_size + output_size
    genome_size = (total_size - input_size)*total_size + bias_size

    genome = G1DList.G1DList(genome_size)
    genome.setParams(rangemin=-3, rangemax=3)
    genome.setParams(input_size=input_size,  hidden_size=hidden_size,  output_size=output_size)
    genome.setParams(gauss_mu=0., gauss_sigma=1.0)

    genome.initializator.set(Initializators.G1DListInitializatorReal)
    genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
    genome.evaluator.set(eval_func)
    genome.crossover.set(Crossovers.G1DListCrossoverSinglePoint)

    ga = GSimpleGA.GSimpleGA(genome)
#    print "STOPPING ELITISM"
    ga.setElitism(True)
    ga.setElitismReplacement(1)

#    ga.selector.set(Selectors.GTournamentSelectorAlternative)
#    ga.selector.set(Selectors.GRankSelector)
    ga.selector.set(Selectors.GRouletteWheel)
    
    ga.setGenerations(10000)
    ga.setPopulationSize(800)
    ga.setCrossoverRate(0.0)
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
