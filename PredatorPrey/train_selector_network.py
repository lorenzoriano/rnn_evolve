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

nsensors = 8
DIR = "/home/pezzotto/Logs/PredatorPrey/"

file = open(DIR + "Hidden15PredatorCircle/bestnet.txt", "r")
prey_net = pickle.load(file)
file.close()
file = open(DIR + "ComplexNet/bestnet.txt", "r")
complex_net = pickle.load(file)
file.close()


def chromosome_convert(chromosome):
    input_size = chromosome.getParam("input_size")
    output_size = chromosome.getParam("output_size")
    hidden_size = chromosome.getParam("hidden_size")
    bias_size = hidden_size + output_size

    net = variable_rnn.SelectorRNN(hidden_size, prey_net,  complex_net)
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
    trials = 20

    times = []
    for i in xrange(trials):
        world = competitive_scenario.World()
        elapsed_time = 0
        predator_vw = competitive_scenario.predator_lookahead_strategy(buffer_size=10,  lookahead_steps = 10)

        while not world.collision() and elapsed_time < max_time:
#            v_predator = competitive_scenario.predator_greedy_strategy(predator_speed,  world)
            v_predator = predator_vw(predator_speed,  world)
#            sensor_input = competitive_scenario.create_prey_sensory_inputs_vel(world,  nsensors,  prey_speed)
            sensor_input = competitive_scenario.create_prey_sensory_inputs(world,  nsensors)

            input_arr = np.array(sensor_input,  ndmin=2)
            netout = net(input_arr)

            v = netout[0] * prey_speed
            w = 2.*np.pi*netout[1] - np.pi
            v_prey = (v,  w)

            world.update(v_prey,  v_predator,  dt)
            elapsed_time = elapsed_time + dt

        times.append(elapsed_time)

    mn = np.median(times)
    return mn

def eval_func(chromosome,  **args):
    net = chromosome_convert(chromosome)
    return calculate_fitness(net)

# Genome instance
hidden_size = 5
fake_net = variable_rnn.SelectorRNN(hidden_size,  prey_net,  complex_net)

input_size = fake_net.input_size
output_size = fake_net.output_size

print "Fake Net: ",  input_size,  "\t",  hidden_size,  "\t",  output_size

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
ga.setGenerations(2000)
ga.setPopulationSize(200)
ga.setCrossoverRate(0.0)
ga.setMutationRate(0.8)
ga.setMinimax(Consts.minimaxType["maximize"])
sqlite_adapter = DBAdapters.DBSQLite(dbname=DIR+"evolution.db", identify="pippo")
ga.setDBAdapter(sqlite_adapter)

# Do the evolution
ga.evolve(freq_stats = 10)

# Best individual
best = ga.bestIndividual()
net = chromosome_convert(best)
#net.to_dot("/home/pezzotto/Logs/RobotFollowing/net.dot")

pickle_file = open(DIR+"bestnet.txt", "w")
print "Pickling"
pickle.dump(net,  pickle_file, 2)


print "Done"
