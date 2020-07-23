# See example (GA) onemax_ray.py
# See example (GP) symbreg_ray.py

# Derek M Tishler
# Jul 23 2020

'''
A replacement for map using Ray that automates batching. Works on partials as to allow and fix pickle issues such as
DeltaPenalty and addEphemeralConstant when working at scale(many processes on machine or cluster of nodes)
'''

from time import sleep

import ray
from ray.util import ActorPool


@ray.remote
class Ray_Deap_Map():
    def __init__(self, creator_setup=None, pset_creator=None):
        # issue 946? Ensure non trivial startup to prevent bad load balance across a cluster
        sleep(0.1)

        # recreate scope from global (for ex need toolbox in gp too)
        self.creator_setup = creator_setup
        if creator_setup is not None:
            self.creator_setup()
        self.pset_creator = pset_creator
        if pset_creator is not None:
            self.pset_creator()

    def ray_remote_eval_batch(self, zipped_input):
        f, iterable, id_ = zipped_input
        return [(f(i), id_) for i in iterable]

# if eval is inxpensive, ray will be slow at scale(network overhead) just like scoop. If that is the case need to batch out work
@ray.remote
class Ray_Deap_Map_Manager():
    def __init__(self, creator_setup=None, pset_creator=None):
        # issue 946? Ensure non trivial startup to prevent bad load balance across a cluster
        sleep(0.1)

        # recreate scope from global (for ex need toolbox in gp too)
        self.creator_setup = creator_setup
        if creator_setup is not None:
            self.creator_setup()
        self.pset_creator = pset_creator
        if pset_creator is not None:
            self.pset_creator()

        self.n_workers = int(ray.cluster_resources()['CPU'])

    def map(self, func, iterable):

        if self.n_workers == 1:
            # only 1 worker, normal list comp
            results = [func(item) for item in iterable]
        else:
            # many workers, lets use ActorPool

            if len(iterable) < self.n_workers:
                n_workers = len(iterable)
            else:
                n_workers = self.n_workers

            n_per_batch = int(len(iterable)/n_workers) + 1
            batches = [iterable[i:i + n_per_batch] for i in range(0, len(iterable), n_per_batch)]
            id_for_reorder = range(len(batches))

            eval_pool = ActorPool([Ray_Deap_Map.remote(self.creator_setup, self.pset_creator) for _ in range(n_workers)])
            unordered_results = list(eval_pool.map_unordered(lambda actor, input_tuple: actor.ray_remote_eval_batch.remote(input_tuple),
                                                             zip([func]*n_workers, batches, id_for_reorder)))
            
            # ensure order of batches
            ordered_batch_results = [batch for batch_id in id_for_reorder for batch in unordered_results if batch_id == batch[0][1]]
            
            #flatten batches to list of fitnes
            results = [item[0] for sublist in ordered_batch_results for item in sublist]
            
        return results

# This is what we register for map in deap scripts
# we need to explicitly launch .remote() on ray workers, so cant assign as partiels like scoop and pool in old days
def ray_deap_map(func, pop, creator_setup=None, pset_creator=None):
    map_worker = Ray_Deap_Map_Manager.remote(creator_setup, pset_creator)

    results = ray.get(map_worker.map.remote(func, pop))

    return results
