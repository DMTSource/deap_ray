# Based on:
# https://github.com/DEAP/deap/blob/master/examples/gp/symbreg.py

# Derek M Tishler
# Jul 2020

import operator
import math
import random

import numpy
numpy.random.seed(318)

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import gc

## Ray init code, user needs to apply#################
# see: https://docs.ray.io/en/master/walkthrough.html
import ray
from ray_map import ray_deap_map

# 1 will use normal map on main(useful for debugging?), above 1 will use ActorPool and batching.
num_cpus = 4

# Setting resources(memory, object_store_memory) helps to ensure consistent results.
# View the Ray dashboard at localhost:8265 for more info if unsure(start with plain ray.init()))
ray.init(num_cpus=num_cpus,
         memory=num_cpus * 1.3 * 1073741824, # n_cpu * 0.5gb for memory
         object_store_memory=num_cpus * 0.3 * 1073741824) # n_cpu * 0.25gb for object store

'''
'time python symbreg_ray.py' on my machine(8 processors) shows:
num_cpus=1 (map): 0m19.575s (real)
num_cpus=2 (ray): 0m13.447s (real)
num_cpus=3 (ray): 0m11.672s (real)
num_cpus=4 (ray): 0m10.704s (real)
num_cpus=5 (ray): 0m9.515s (real)
num_cpus=6 (ray): 0m11.135s (real)
num_cpus=7 (ray): 0m11.252s (real)
num_cpus=8 (ray): 0m11.301s (real)
'''
######################################################


# Define new functions
def protectedDiv(left, right):
    if right == 0.:
        return 0.
    try:
        return left / right
    except ZeroDivisionError:
        return 1.


##This is different!#################################
def pset_creator():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    # we only have to form and pass pset_creator to ray workers because
    #    this singe item, else cant be found in sscope
    pset.addEphemeralConstant("rand101", lambda: random.uniform(-1,1))
    pset.renameArguments(ARG0='x')
    return pset
pset = pset_creator()

## GP+Ray Requires both creators so we can compile inside val as most examples show
def creator_setup():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
creator_setup()
######################################################

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

######################################################
# use a shared memory object to prevent copy of data to each eval
#points = numpy.random.uniform(-1., 1., (int(1e3),)) #not much speedup at all
points = numpy.random.uniform(-1., 1., (int(1e4),)) #now we see it scale

shared_points_store_id = ray.put(points)

del points
gc.collect()
######################################################

def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    ######################################################
    shared_memory_points = ray.get(shared_points_store_id)
    ######################################################

    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in shared_memory_points)

    return math.fsum(sqerrors) / float(len(shared_memory_points)),

toolbox.register("evaluate", evalSymbReg) 
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

##This is different!#################################
toolbox.register("map", ray_deap_map, creator_setup=creator_setup, pset_creator=pset_creator)
######################################################

def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 3, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

if __name__ == "__main__":
    main()
