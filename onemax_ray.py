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
#ray.init(num_cpus=1)
ray.init()

# THEN import the tool
from ray_map import ray_deap_map
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
    return sum(individual),

# from https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html
def feasible(individual):
    """Feasibility function for the individual. Returns True if feasible False
    otherwise."""
    if 50 < individual[0] < 60:
        return True
    return False

# from https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html
def distance(individual):
    """A distance function to the feasibility region."""
    return (individual[0] - 5.0)**2


toolbox.register("evaluate", evalOneMax)
# Here we apply a feasible constraint: 
# https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 50.0, distance))
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

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, 
                        stats=stats, halloffame=hof)
