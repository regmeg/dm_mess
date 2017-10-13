'''
This module handles the parameter definitions of the model
'''
import tensorflow as tf
import numpy as np
import random
import time
import os
import sys
import datetime


tf.flags.DEFINE_integer("epochs", 1000000, "num_of_epochs")    
tf.flags.DEFINE_integer("num_batches", 4, "num_batches")    
tf.flags.DEFINE_float("test_ratio", 0.3, "test_ratio")
tf.flags.DEFINE_integer("test_cycle", 100, "test_cycle")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
tf.flags.DEFINE_boolean("sel_col", True, "whether sel confined columns")
tf.flags.DEFINE_integer("lay_1", 5, "lay_1")
tf.flags.DEFINE_integer("lay_2", 3, "lay_2")
tf.flags.DEFINE_float("drop_rate", 0.35, "drop_rate")
tf.flags.DEFINE_boolean("stochastic", True, "whether to shuffle samples")
tf.flags.DEFINE_boolean("add_noise", False, "whether to add gradient noise")
tf.flags.DEFINE_boolean("norm", True, "whether to norm grads")
tf.flags.DEFINE_integer("seed", round(random.random()*100000), "the global simulation seed for np and tf")
tf.flags.DEFINE_string("name", "predef_sim_name" , "name of the simulation")
tf.flags.DEFINE_boolean("logoff", False , "stitch of loggin")

FLAGS = tf.flags.FLAGS

def get_cfg():
    #configuraion constants
    global_cfg = dict(
        sim_start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        dtype_np = np.float64,
        dtype_tf = tf.float64,

        epochs = FLAGS.epochs,
        num_batches = FLAGS.num_batches,
        test_ratio = FLAGS.test_ratio,
        test_cycle = FLAGS.test_cycle,
        learning_rate = FLAGS.learning_rate,
        sel_col = FLAGS.sel_col,
        lay_1 = FLAGS.lay_1,
        lay_2 = FLAGS.lay_2,
        drop_rate = FLAGS.drop_rate,
        stochastic = FLAGS.stochastic,
        add_noise = FLAGS.add_noise,
        norm = FLAGS.norm,
        seed = FLAGS.seed,
        name = FLAGS.name,
        logoff = FLAGS.logoff
    )
    global_cfg['dst'] =  global_cfg['name']

    
    
    return global_cfg