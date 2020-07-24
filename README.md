# deap_ray
Ray based map function for Deap with examples. Replaces SCOOP for multiprocess/cluster use in Deap for Python 3.  
Automatically batches iterator to multiple workers for many processors or clusters.  

This repository is temporary, ideally, and can be added to Deap once python 3 is standardized as Ray requires python > 3.5:  
https://github.com/DEAP/deap/issues/290 (Commit to py3)  
https://github.com/DEAP/deap/issues/404 (Replace Scoop)  

Ray is an amazing tool, not just for reinforcement learning, but for general compute workloads in python. With SCOOP no longer supported, it seems Ray one of the best solutions for Deap to scale to clusters. I have created this tool to, like Scoop and multiprocess before it, drop into Deap scripts via a simple map-like call in the toolbox to enable use of Ray to scale a heavy workload.

Ray + Deap gives some major benefits, such as decorators like DeltaPenalty and Ephemeral Constants (addEphemeralConstant) working remotely and shared memory objects allowing for even bigger scale if memory is a bottleneck. Before, these issues were extremely problematic in parallel solutions as they could not be pickled in some cases.

Please check out Ray, its great!  
https://docs.ray.io/en/master/walkthrough.html  
https://github.com/ray-project/ray  

Please check out Deap, its great!  
https://deap.readthedocs.io/en/master/  
https://github.com/DEAP/deap  

This repo includes the file you will need to import to your Deap script:  
**ray_map.py**

Also included are two simple, Genetic Programming(GP) and Genetic Algorithm(GA), examples to illustrate the nuances of implementation in your own Deap code. I attempted to make switching to parallel, like the previous uses of SCOOP and MP, as simple as possible with Deap. I will work to improve this code further as some use cases are probably missing. Please report issues!  
**onemax_ray.py** *(Uses a DeltaPenalty to illustrate power of Ray to properly prevent pickle problems!)*  
**symbreg_ray.py** *(Uses an addEphemeralConstant to illustrate power of Ray to penultimately persuade pickle predicaments!)*  

###### Note: 
Much like scoop, there is no magic bullet to fight network overheard and spooling up remote workers. In these examples, evaluation of individuals is a fast, trivial task. You will likely notice a slowdown in smaller parallel loads vs the standard examples on 1 process. This code automatically and evenly batches out the passed along iterable, if ray.init() utilizes 1 or more processes, to better utilize each remote worker. 

Don't forget to check out Ray's shared memory objects vs loading things inside evaluation functions. This can drastically reduce memory per remote worker! Use of ray.put / ray.get for shared memory items can he found here:  
https://docs.ray.io/en/master/walkthrough.html?highlight=put

###### About:
I started using Deap for Medical Physics research and optimizations many years ago. Recently, I began using Ray and it has helped me scale python code in ways I can't describe(yes I can, see shared memory objects above). I hope this code may be of use to others who enjoy using Deap for fun and work!  
https://www.linkedin.com/in/derekmtishler/
