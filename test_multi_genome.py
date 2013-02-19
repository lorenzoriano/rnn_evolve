from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators,  Consts
from pyevolve import DBAdapters
from pyevolve import Crossovers

from rnn_evolve import multi_chromosome

import random

def eval_func(chromosome,  **args):
    
    realgen = chromosome.real_genome
    bingen = chromosome.binary_genome
    
    return sum(realgen.genomeList) + sum(bingen.genomeList)

if __name__ == "__main__":
    # Genome instance
    realsize = 3
    binsize = 3
    
    genome = multi_chromosome.RealBin(realsize,  binsize)
    genome.real_genome.setParams(rangemin=0, rangemax=1)
    genome.real_genome.setParams(gauss_mu=0., gauss_sigma=0.1)
    genome.real_genome.initializator.set(Initializators.G1DListInitializatorReal)
    genome.real_genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
    
    genome.binary_genome.mutator.set(Mutators.G1DBinaryStringMutatorFlip)
    
    genome.evaluator.set(eval_func)
#    genome.crossover.set(Crossovers.G1DListCrossoverSinglePoint)

    ga = GSimpleGA.GSimpleGA(genome)
    print "STOPPING ELITISM"
    ga.setElitism(False)
#    ga.setElitismReplacement(1)

#    ga.selector.set(Selectors.GTournamentSelectorAlternative)
#    ga.selector.set(Selectors.GRankSelector)
    ga.selector.set(Selectors.GRouletteWheel)
    
    ga.setGenerations(100)
    ga.setPopulationSize(20)
    ga.setCrossoverRate(0.5)
    ga.setMutationRate(0.8)
    ga.setMinimax(Consts.minimaxType["maximize"])

    # Do the evolution
    ga.evolve(freq_stats = 10)

    # Best individual
    best = ga.bestIndividual()
    print best


    print "Done"
    
