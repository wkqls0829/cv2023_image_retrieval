from batchminer import random_distance, intra_random
from batchminer import lifted, rho_distance, softhard, npair, parametric, random, semihard, distance, softhard_balanced

BATCHMINING_METHODS = {'random':random,
                       'semihard':semihard,
                       'softhard':softhard,
                       'distance':distance,
                       'rho_distance':rho_distance,
                       'npair':npair,
                       'parametric':parametric,
                       'lifted':lifted,
                       'random_distance': random_distance,
                       'intra_random': intra_random,
                       'softhard_balanced': softhard_balanced,}


def select(batchminername, opt):
    #####
    if batchminername not in BATCHMINING_METHODS: raise NotImplementedError('Batchmining {} not available!'.format(batchminername))

    batchmine_lib = BATCHMINING_METHODS[batchminername]

    return batchmine_lib.BatchMiner(opt)
