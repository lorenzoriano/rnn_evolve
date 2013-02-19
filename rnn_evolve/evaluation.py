import numpy as np

def evaluate_net(individual,  **args):
    if  not individual.mutable:
        return individual.fitness
    net = individual.RNN
    xs = args["xs"]
    ys = args["ys"]
    sum_squared_err = 0
    for x in xs:
        net_out = net(x)
        sum_squared_err += np.linalg.norm(ys - net_out)**2

    return -sum_squared_err

def calculate_offsprings(vs,  **args):
    """
    vs: the fitness values
    """
    gamma = args.get("gamma",  1.0)
    vmax = np.max(vs)
    sigma2 = np.std(vs)
    h = np.exp(-gamma/sigma2 * (vmax - vs))

    lam = len(vs)/np.sum(h) * h
    m = np.array([np.random.poisson(l) for l in lam])
    return m

def evaluate_population(population, **args):
    fitness = [evaluate_net(individual, **args) for individual in population]
    ms = calculate_offsprings(np.array(fitness) ,  **args)

    avg = np.mean(fitness)
    mx = np.max(fitness)
    mn = np.min(fitness)
    if args.get("display_stats", False):
        print "Fitness: Max ",mx ,  " Min: ",mn  ,  " Avg: ",  avg

    for m, f, individual in zip(ms, fitness, population):
        individual.offspring = m
        individual.fitness = f

    return mx, mn, avg

