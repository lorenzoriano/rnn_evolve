from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators,  Consts
from pyevolve import DBAdapters
from pyevolve import Crossovers

import numpy as np
import pickle

import rnn_evolve.variable_rnn as variable_rnn
import competitive_scenario

prey_file = open("/home/pezzotto/Logs/PredatorPrey/PreyZeroHidden/bestnet.txt", "r")
prey_net = pickle.load(prey_file)
prey_file.close()

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
    trials = 10

    times = []
    for i in xrange(trials):
        world = competitive_scenario.World()
        elapsed_time = 0
        while not world.collision() and elapsed_time < max_time:
            #predator
            sensor_input = competitive_scenario.create_predator_sensory_inputs(world)
            input_arr = np.array(sensor_input,  ndmin=2)
            netout = net(input_arr)
            v = netout[0] * predator_speed
            w = 2.*np.pi*netout[1] - np.pi
            v_predator = (v,  w)

            #prey
#            sensor_input = competitive_scenario.create_prey_sensory_inputs(world)
#            input_arr = np.array(sensor_input,  ndmin=2)
#            netout = prey_net(input_arr)
#            v = netout[0] * prey_speed
#            w = 2.*np.pi*netout[1] - np.pi
#            v_prey = (v,  w)
            v_prey = (0,  0)

            world.update(v_prey,  v_predator,  dt)
            elapsed_time = elapsed_time + dt

        times.append(elapsed_time)

    mn = np.median(times)
#    if np.abs(mn - 120.10)<1:
#        print ",Median: ", mn,  "\tArr: ",  times
    return mn


def eval_func(chromosome,  **args):
    net = chromosome_convert(chromosome)
    return calculate_fitness(net)


# Genome instance
input_size = 2
hidden_size = 5
output_size = 2
bias_size = hidden_size + output_size
total_size = input_size + hidden_size + output_size
genome_size = (total_size - input_size)*total_size + bias_size

genome = G1DList.G1DList(genome_size)
genome.setParams(rangemin=-1, rangemax=1)
genome.setParams(input_size=input_size,  hidden_size=hidden_size,  output_size=output_size)
genome.setParams(gauss_mu=0., gauss_sigma=0.1)

genome.initializator.set(Initializators.G1DListInitializatorReal)
genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
genome.evaluator.set(eval_func)
genome.crossover.set(Crossovers.G1DListCrossoverTwoPoint)

ga = GSimpleGA.GSimpleGA(genome)
ga.selector.set(Selectors.GRouletteWheel)
#ga.selector.set(Selectors.GRankSelector)
ga.setGenerations(500)
ga.setPopulationSize(100)
ga.setCrossoverRate(0.2)
ga.setMutationRate(0.8)
ga.setMinimax(Consts.minimaxType["minimize"])
#sqlite_adapter = DBAdapters.DBSQLite(dbname="/home/pezzotto/Logs/RobotFollowing/evolution.db", identify="hidden-10")
#ga.setDBAdapter(sqlite_adapter)

# Do the evolution
ga.evolve(freq_stats = 10)

# Best individual
best = ga.bestIndividual()
net = chromosome_convert(best)
#net.to_dot("/home/pezzotto/Logs/RobotFollowing/net.dot")

pickle_file = open("/home/pezzotto/Logs/PredatorPrey/bestnet.txt", "w")
print "Pickling"
pickle.dump(net,  pickle_file, 2)



print "Done"
