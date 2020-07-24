# Based on:
# https://github.com/DEAP/deap/blob/master/examples/gp/symbreg.py

# Derek M Tishler
# Jul 2020

import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

## Ray init code, user needs to apply#################
# see: https://docs.ray.io/en/master/walkthrough.html
import ray
from ray_map import ray_deap_map

#ray.init(num_cpus=1) # will use default python map on current process, useful for debugging?
ray.init(num_cpus=4) # over 1 assigned proc will batch out via ActorPool

'''
Eval is made arbitrarily more expensive to show differnce.
'time python symbreg_ray.py' on my machine(8 processors) shows:
num_cpus=1 (map): 20.3 sec(real)
num_cpus=2 (ray): 14.1 sec(real)
num_cpus=4 (ray): 11.9 sec(real)
num_cpus=7 (ray): 13.1 sec(real)
num_cpus=8 (ray): 13.0 sec(real)
'''
######################################################



# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


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
    pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
    pset.renameArguments(ARG0='x')
    return pset
pset = pset_creator()

## GP Requires both creators so we can compile inside val as most examples show
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

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # make eval arbitrarily more expensive to illustate ray vs std map
    for _ in range(100):
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x
        sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
        fitness = math.fsum(sqerrors) / float(len(points)),

    return fitness

toolbox.register("evaluate", evalSymbReg, points=[x/100. for x in range(-100,100)]) 
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

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 5, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

if __name__ == "__main__":
    main()
