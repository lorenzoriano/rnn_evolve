from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators,  Consts
from pyevolve import DBAdapters
from pyevolve import Crossovers

import numpy as np
import pickle

import rnn_evolve.variable_rnn as variable_rnn
from rnn_evolve.c_rnn import c_eval_func

#logistic_x = np.zeros(100)
#logistic_x[0] = 0.5
#for i in xrange(1,len(logistic_x)):
#    logistic_x[i] = 3.5*logistic_x[i-1]*(1-logistic_x[i-1])

data = np.loadtxt("/home/pezzotto/Logs/RobotFollowing/followingdata_norm.txt")

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

#def eval_func(chromosome,  **args):
#    net = chromosome_convert(chromosome)
#
#    xs = data[:, :3]
#    ys = data[:, 3:]
#
##    xs = logistic_x[:-1]
##    ys = logistic_x[1:]
##    xs = np.random.uniform(-1., 1.,  (50, 2))
##    ys =np.sin(xs)
##    ys = xs
##    ys = .5 * (xs[:, 0] + xs[:, 1])
#    sum_squared_err = 0
#    for x, y in zip(xs, ys):
#        net_out = net(x).ravel()
##        sum_squared_err += np.abs(y - net_out[0])
#        sum_squared_err += np.linalg.norm(y - net_out)
#
#    return sum_squared_err

def eval_func(chromosome,  **args):
    net = chromosome_convert(chromosome)

    xs = data[:, :3]
    ys = data[:, 3:]

    return c_eval_func(xs, ys, net)

# Genome instance
input_size = 3
hidden_size = 10
output_size = 2
bias_size = hidden_size + output_size
total_size = input_size + hidden_size + output_size
genome_size = (total_size - input_size)*total_size + bias_size

genome = G1DList.G1DList(genome_size)
genome.setParams(rangemin=-1, rangemax=1)
genome.setParams(input_size=input_size,  hidden_size=hidden_size,  output_size=output_size)
genome.setParams(gauss_mu=0., gauss_sigma=0.2)

genome.initializator.set(Initializators.G1DListInitializatorReal)
genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
genome.evaluator.set(eval_func)
genome.crossover.set(Crossovers.G1DListCrossoverTwoPoint)

ga = GSimpleGA.GSimpleGA(genome)
#ga.selector.set(Selectors.GRouletteWheel)
ga.selector.set(Selectors.GRankSelector)
ga.setGenerations(1200)
ga.setPopulationSize(1200)
ga.setCrossoverRate(0.2)
ga.setMutationRate(0.2)
ga.setMinimax(Consts.minimaxType["minimize"])
sqlite_adapter = DBAdapters.DBSQLite(dbname="/home/pezzotto/Logs/RobotFollowing/evolution.db", identify="hidden-10")
ga.setDBAdapter(sqlite_adapter)

# Do the evolution
ga.evolve(freq_stats = 10)

# Best individual
best = ga.bestIndividual()
print best
net = chromosome_convert(best)
net.to_dot("/home/pezzotto/Logs/RobotFollowing/net.dot")

pickle_file = open("/home/pezzotto/Logs/RobotFollowing/best.txt", "w")
pickle.dump(net,  pickle_file, 0)
pickle_file.close()

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
