from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators,  Consts
from pyevolve import DBAdapters
from pyevolve import Crossovers

import numpy as np
import cPickle
import sys

import rnn_evolve.variable_rnn as variable_rnn
import competitive_scenario

nsensors = 6
DIR = "/home/pezzotto/Logs/PredatorPrey/FixedWorld/"
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

def calculate_fitness(net):
    dt = 0.1
    max_time = 120

    prey_speed = 0.4
    predator_speed = 0.45
    trials = 4
    distance = 5
    buffer_size=5
    lookahead_steps = 5

    times = []
    for i in xrange(trials):
        world = competitive_scenario.World()
        if i==0:
            world.fixed_distance_top_left(distance)
        elif i==1:
            world.fixed_distance_top_right(distance)
        elif i==2:
            world.fixed_distance_bottom_left(distance)
        elif i==3:
            world.fixed_distance_bottom_right(distance)
        else:
            raise ValueError("Invalid trial: " + str(i))

        elapsed_time = 0
        predator_vw = competitive_scenario.predator_lookahead_strategy(buffer_size=buffer_size,  lookahead_steps = lookahead_steps)

        while (not world.collision()) and (elapsed_time < max_time):
            v_predator = predator_vw(predator_speed,  world)
            sensor_input = competitive_scenario.create_prey_sensory_inputs(world,  nsensors)

            input_arr = np.array(sensor_input,  ndmin=2)
            netout = net(input_arr)

            v = prey_speed
            w = 2.*np.pi*netout[0,0] - np.pi
            v_prey = (v,  w)

            world.update(v_prey,  v_predator,  dt)
            elapsed_time = elapsed_time + dt

        times.append(elapsed_time)

    mn = np.mean(times)
    return mn

def eval_func(chromosome,  **args):
    net = chromosome_convert(chromosome)
    return calculate_fitness(net)

def callback(ga):
    if not (ga.getCurrentGeneration() % 10):
        sys.stdout.write("Dumping... ")
        best = ga.bestIndividual()
        net = chromosome_convert(best)
        pickle_file = open(DIR+"bestnet.txt", "w")
        cPickle.dump(net,  pickle_file, 2)
        pickle_file.close()
        sys.stdout.write("done!\n")
    return False

# Genome instance
input_size = nsensors
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
ga.selector.set(Selectors.GRouletteWheel)
ga.setElitism(True)
#ga.selector.set(Selectors.GRankSelector)
ga.setGenerations(2000)
ga.setPopulationSize(100)
ga.setCrossoverRate(0.2)
ga.setMutationRate(0.8)
ga.setMinimax(Consts.minimaxType["maximize"])
sqlite_adapter = DBAdapters.DBSQLite(dbname=DIR+"evolution.db", identify="pippo")
ga.setDBAdapter(sqlite_adapter)
ga.stepCallback.set(callback)
ga.setElitism(True)

# Do the evolution
ga.evolve(freq_stats = 10)

# Best individual
best = ga.bestIndividual()
net = chromosome_convert(best)

pickle_file = open(DIR+"bestnet.txt", "w")
print "Pickling"
cPickle.dump(net,  pickle_file, 2)
pickle_file.close()

print "Done"
