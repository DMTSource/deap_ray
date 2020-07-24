# Based on:
# https://github.com/DEAP/deap/blob/master/examples/ga/onemax.py

# Derek M Tishler
# Jul 2020

import array
import random
import sys

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

##Ray init code, user needs to apply#################
# see: https://docs.ray.io/en/master/walkthrough.html
import ray
from ray_map import ray_deap_map

#ray.init(num_cpus=1) # will use default python map on current process, useful for debugging?
ray.init(num_cpus=4) # will batch out via ActorPool, slower vs above for trivial loads because overhead

'''
Eval is made arbitrarily more expensive to show difference. Tricky as DeltaPenalty skips evals sometimes.
'time python onemax_ray.py' on my machine(8 processors) shows:
num_cpus=1 (map): 25.5 sec(real)
num_cpus=2 (ray): 17.5 sec(real)
num_cpus=4 (ray): 13.0 sec(real)
num_cpus=7 (ray): 13.3 sec(real)
num_cpus=8 (ray): 13.6 sec(real)
'''
######################################################


##Example code updated, user needs to apply##########
def creator_setup():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
# make sure to call locally
creator_setup()
######################################################


toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Including the problematic DeltaPenalty to illustate ray strength
def evalOneMax(individual):
    # make eval arbitrarily more expensive to illustate ray vs std map
    for _ in range(20000):
        fitness = sum(individual)**2/len(individual),
    return fitness

# from https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html
def feasible(individual):
    """Feasibility function for the individual. Returns True if feasible False
    otherwise."""
    if -0.5 < individual[0] < 0.5:
        return True
    return False

# from https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html
def distance(individual):
    """A distance function to the feasibility region."""
    return (individual[0] - 0.0)**2


toolbox.register("evaluate", evalOneMax)
# Here we apply a feasible constraint: 
# https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 1.0, distance))
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

##This is different!#################################
toolbox.register("map", ray_deap_map, creator_setup = creator_setup)
######################################################

if __name__ == "__main__":

    random.seed(64)
    
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, 
                        stats=stats, halloffame=hof)
