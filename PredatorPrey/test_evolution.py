import rnn_evolve.evolution as evolution
from rnn_evolve.population import generate_random_population
import rnn_evolve.population as population
import rnn_evolve.variate as variate
import rnn_evolve.evaluation as evaluation
import rnn_evolve.reproduction as reproduction
import rnn_evolve.variable_rnn as variable_rnn

import numpy as np
import unittest

class TestEvolution(unittest.TestCase):
    def setUp(self):
        self.params = {
                  "neurons_delete_prob" : 0.01,
                  "neuron_insert_prob" : 0.01,
                  "neurons_modify_prob" : 0.05,
                  "synapses_delete_prob" : 0.05,
                  "synapses_insert_prob" : 0.05,
                  "synapses_modify_prob" : 0.05,
                  "neurons_sigma" : 0.1,
                  "synapses_sigma"  : 0.1,
                  "maxpop":  3000,
                  "minpop":  2000,
                  "display_stats":  True,
                  "gamma": 1.0,
                  "max_steps": 300,
                  "display_interval":  1,
                  "input_size":  1,
                  "output_size": 1,
                  "hidden_size":  3
                  }

        xs = np.random.uniform(-1., 1.,  100)
        ys = np.sin(xs)

        self.params["xs"] = xs
        self.params["ys"] = ys

        self.population = generate_random_population(self.params["minpop"], True, **self.params)
#        right_net = variable_rnn.VariableRNN(0, 1, 1)
#        right_net.W[1, 0] = 0.5
#        self.population = [population.Individual(right_net)] + self.population


    def __evaluate_net(self, individual,  **args):
        net = individual.RNN
#        xs = args["xs"]
#        ys = args["ys"]
        xs = np.random.uniform(-1., 1.,  100)
        ys =np.sin(xs)
        sum_squared_err = 0
        for x, y in zip(xs, ys):
            net_out = net(x)
            self.assertTrue(np.isfinite(net_out))
            sum_squared_err += np.linalg.norm(y - net_out)**2

        return -sum_squared_err

    def __evaluate_population(self, population, **args):
        fitness = [self.__evaluate_net(individual, **args) for individual in population]
        ms = evaluation.calculate_offsprings(np.array(fitness) ,  **args)

        avg = np.mean(fitness)
        mx = np.max(fitness)
        mn = np.min(fitness)
        if args.get("display_stats", False):
            print "Fitness: Max ",mx ,  " Min: ",mn  ,  " Avg: ",  avg

        for m, f, individual in zip(ms, fitness, population):
            individual.offspring = m
            individual.fitness = f

        return mx, mn, avg


    def test_evolution(self):
        args = self.params
        pop = self.population

        max_steps = args["max_steps"]
        display_steps = args.get("display_interval", 1)
        display_stats = args.get("display_stats", False)

        coll_data = []

        for t in xrange(max_steps):
            print "\nGeneration ",  t+1,  "Population size: ",  len(pop)
            mutated = 0
            for individual in pop:
                if individual.mutable:
                    mutated += 1
                variate.mutate(individual)
            print "Mutated: ", mutated

            (mx, mn, avg) = self.__evaluate_population(pop, **args)
            coll_data.append([mx, mn, avg])
            pop = reproduction.reproduce(pop, **args)
            print "Check this (mutable): ",  np.sum([individual.mutable for individual in pop])


            mutated = 0

        self.population = pop
        best_individual = population.find_maximum_individual(pop)
        print best_individual
        xs = args["xs"]
        ys = args["ys"]
        for x, y in zip(xs, ys):
            net_out =  best_individual.RNN(x)[0][0]
            print "x: ", x,  " -> real: ", y, "\tnet: ",net_out,  "\tdiff: ", y - net_out,  "\t sse: ",  np.linalg.norm(y - net_out)**2
        np.savetxt("/home/pezzotto/gen.txt",  coll_data)



if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEvolution)
#    unittest.TextTestRunner(verbosity=2).run(suite)
    suite.debug()
