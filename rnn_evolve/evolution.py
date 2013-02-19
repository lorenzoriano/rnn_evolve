import variable_rnn
import variate
import evaluation
import reproduction

def evolution_step(population,  **args):
    for individual in population:
        variate.mutate(individual.RNN)

    evaluation.evaluate_population(population, **args)
    population = reproduction.reproduce(population, **args)
    return population

def whole_evolution(population,  **args):
    max_steps = args["max_steps"]
    display_steps = args.get("display_interval", 1)
    display_stats = args.get("display_stats", False)

    for t in xrange(max_steps):
        population = evolution_step(population, **args)
        if not t % display_steps:
            print "Generation ",  t,  "Population size: ",  len(population)
