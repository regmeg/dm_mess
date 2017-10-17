'''
This module simulates the gridsearch funtionality, in order to tune the hyperparmaters
'''
import os
import subprocess
import sys
import random
import time
import tensorflow as tf
import itertools
from collections import OrderedDict

tf.flags.DEFINE_integer("seed", 0, "the global simulation seed for np and tf")
tf.flags.DEFINE_string("type", " ", "model type")

FLAGS = tf.flags.FLAGS

def gen_cmd(cfg_dict, seed):
    string = "python3 ./regression.py"
    name = " --name="
    for key, val in cfg_dict.items():
        if key == 'lay':
            for i,v in enumerate(val):
                string += " --"+key+"_"+str(i+1)+"="+str(v)
                name += key+"_"+str(i+1)+"#"+str(v)+"~"
        else:
            string += " --"+str(key)+"="+str(val)
            name += str(key)+"#"+str(val)+"~"
    if seed == 0: seed = int(round(random.random()*100000))
    name += "seed#" + str(seed)
    seed  = " --seed="+str(seed)
    return string + seed + name



params=OrderedDict()
params['num_batches'] = [2]
params['learning_rate'] = [0.001]
params['sel_col'] = [False]
params['lay'] = [[8], [5,3], [5,3,2]]
params['stochastic'] = [True]
params['drop_rate'] = [0.35, 0.5]
params['add_noise'] = [False]
params['norm'] = [True]


#seed
seed = FLAGS.seed
#for n in range(len(cfg)):
cfg_dicts = [OrderedDict(zip(params, x)) for x in itertools.product(*params.values())]

cmds = [gen_cmd(cdict, seed) for cdict in cfg_dicts]

for ind,cmd in enumerate(cmds):
    print("Lnch[" + str(ind+1) +"]: " + cmd)
    subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)