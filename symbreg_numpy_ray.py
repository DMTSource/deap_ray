# Based on:
# https://raw.githubusercontent.com/DEAP/deap/master/examples/gp/symbreg_numpy.py

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
num_cpus=1 (map): 0m15.032s (real)
num_cpus=2 (ray): 0m12.260s (real)
num_cpus=3 (ray): 0m11.061s (real)
num_cpus=4 (ray): 0m10.899s (real)
num_cpus=5 (ray): 0m10.846s (real)
num_cpus=6 (ray): 0m10.967s (real)
num_cpus=7 (ray): 0m11.258s (real)
num_cpus=8 (ray): 0m11.424s (real)
'''
######################################################

# Define new functions
def protectedDiv(left, right):
    with numpy.errstate(divide='ignore',invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1
            x[numpy.isnan(x)] = 1
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 1
    return x

##This is different!#################################
def pset_creator():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(numpy.add, 2, name="vadd")
    pset.addPrimitive(numpy.subtract, 2, name="vsub")
    pset.addPrimitive(numpy.multiply, 2, name="vmul")
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(numpy.negative, 1, name="vneg")
    pset.addPrimitive(numpy.cos, 1, name="vcos")
    pset.addPrimitive(numpy.sin, 1, name="vsin")
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
#samples = numpy.linspace(-1, 1, int(1e5)) #not much speedup at all
samples = numpy.linspace(-1, 1, int(1e6)) #now we see it scale
values = samples**4 + samples**3 + samples**2 + samples

shared_samples_store_id = ray.put(samples)
shared_values_store_id = ray.put(values)

del samples
del values
gc.collect()
######################################################

def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    ######################################################
    shared_memory_samples = ray.get(shared_samples_store_id)
    shared_memory_values = ray.get(shared_values_store_id)
    ######################################################

    # Evaluate the mean squared error between the expression
    # and the real function values : x**4 + x**3 + x**2 + x 
    diff = numpy.sum((func(shared_memory_samples) - shared_memory_values)**2)

    return diff/float(len(shared_memory_values)),

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

##This is different!#################################
toolbox.register("map", ray_deap_map, creator_setup=creator_setup, pset_creator=pset_creator)
######################################################

def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 4, stats, halloffame=hof)

    return pop, stats, hof

if __name__ == "__main__":
    main()
