import population
import variable_rnn
import numpy as np

def reproduce(pop,  **args):

    oldlen = len(pop)
    pop[:] = [individual for individual in pop if individual.offspring > 0]
    print "Number of childless fathers: ",  oldlen - len(pop)

    #generating according to offsprings
    print "Number of immutable fathers (and new pop before children) ",  len(pop)
    newpop = pop[:]
    for individual in pop:
        individual.mutable = False
        #newpop.append(individual)
        for i in xrange(individual.offspring): #sons
#            if len(newpop) >= args["maxpop"]:
#                break
            newborn = individual.clone()
            newborn.mutable = True
            newpop.append(newborn)

    #deleting worst elements
    if len(newpop) >= args["maxpop"]:
        print "Deleting ", len(newpop) - args["maxpop"],  " elements"
        newpop.sort()
        newpop = newpop[:args["maxpop"]]

    #filling with random nets
    if len(newpop) < args["minpop"]:
        "Filling with ",  args["minpop"] - len(newpop), " elements"
        newpop += population.generate_random_population(args["minpop"] - len(newpop), True,  **args)

    print "Check this (not mutable): ",  np.sum([not individual.mutable for individual in newpop])
    print "Number of children: ",  len(newpop) - len(pop)
    return newpop
