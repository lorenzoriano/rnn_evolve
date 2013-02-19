import variable_rnn

class Individual(object):
    def __init__(self, RNN):
        self.offspring = 0
        self.fitness = 0
        self.RNN = RNN
        self.mutable = False

    def __str__(self):
        print "Fitness: ",  self.fitness
        print "Mutable: ",  self.mutable
        print "RNN: ",  self.RNN

    def clone(self):
        newme = Individual(self.RNN.clone())
        newme.fitness = self.fitness
        newme.offspring = self.offspring
        newme.mutable = self.mutable
        return newme

    def __le__(self,  other):
        return self.finess < other.fitness

def generate_random_population(size, mutable=False,  **args):
    input_size = args["input_size"]
    output_size = args["output_size"]
    hidden_size = args["hidden_size"]

    population = []
    for i in xrange(size):
        net = variable_rnn.generate_random_rnn(hidden_size, input_size,  output_size)
        ind = Individual(net)
        ind.mutable = mutable
        population.append(ind)

    return population

def find_maximum_individual(population):
    max = population[0].fitness
    index = 0
    for i,  individual in enumerate(population):
        if individual.fitness > max:
            max = individual.fitness
            index = i
    return population[index]
