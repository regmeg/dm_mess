from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import random_ops

import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn import cross_validation
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from random import randint
import random
from sklearn.utils import shuffle
from numpy import random as random_np


seed = int(round(random.random()*100000))

path_full = "./no_imdb_names-count_cat-tf_184f.csv"
"""
path_train = "../dataset/no_imdb_names-count_cat-tf_184f_train.csv"
path_test = "../dataset/no_imdb_names-count_cat-tf_184f_test.csv"
"""

dta_full = pd.read_csv(path_full)
dta_full = dta_full.fillna(value=0, axis=1)
dta_full = dta_full.dropna()
dta_full = dta_full.drop('Unnamed: 0', axis=1)

"""
dta_train = pd.read_csv(path_train)
dta_train = dta_train.fillna(value=0, axis=1)
dta_train = dta_train.dropna()
dta_train = dta_train.drop('Unnamed: 0', axis=1)

dta_test = pd.read_csv(path_test)
dta_test = dta_test.fillna(value=0, axis=1)
dta_test = dta_test.dropna()
dta_test = dta_test.drop('Unnamed: 0', axis=1)
"""

try:
        os.makedirs('./summaries_movies/'+str(seed))
except FileExistsError as err:
        raise Exception('Dir already exists, saving resultsi n the same dir will result in unreadable graphs')

stdout_org = sys.stdout
sys.stdout = open('./summaries_movies/'+str(seed) + '/log.log', 'w')
        
        
dtype_np = np.float64

##shape = 4812, 1	
y = dta_full['worldwide_gross'].as_matrix().astype(dtype_np)
##shape = 4812, 183
x = dta_full.drop('worldwide_gross', axis=1).as_matrix().astype(dtype_np)





X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x, y, test_size=0.3, random_state=500)

total_len = X_train.shape[0]

print(X_train)
print(X_test)

print(Y_train)
print(Y_test)


##some rules of thumb: https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
#The number of hidden neurons should be between the size of the input layer and the size of the output layer.
#The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer - 2/3*183 + 1 = 123
#The number of hidden neurons should be less than twice the size of the input layer. - 183*2 = 366


# Parameters
learning_rate = 0.001
training_epochs = 1000000
batch_size = 250
display_step = 1
# Network Parameters
n_features = X_train.shape[1]
n_hidden_1 = int(n_features*3) # 1st layer number of features
n_hidden_2 = int(n_features*1.5) # 2nd layer number of features
#n_hidden_3 = 20
#n_hidden_4 = 20
#n_input = 
n_classes = 1

test_cycle = 100
drop_rate = 0.2

# tf Graph input
dtype_tf = tf.float64

xbatch = tf.placeholder(dtype_tf, [None, 183])
ybatch = tf.placeholder(dtype_tf, [None])
train_pl = tf.placeholder(dtype=tf.bool)



# Create model
def multilayer_perceptron(x, params, training):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, params['h1']), params['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.layers.dropout(layer_1, drop_rate, training = training)
    
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, params['h2']), params['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.layers.dropout(layer_2, drop_rate, training = training)

    """
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, params['h3']), params['b3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.layers.dropout(layer_3, drop_rate, training = training)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, params['h4']), params['b4'])
    layer_4 = tf.nn.relu(layer_4)
    layer_4 = tf.layers.dropout(layer_4, drop_rate, training = training)
    """
    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, params['hout']) + params['bout']
    return out_layer


#set seed
tf.set_random_seed(seed)
# Store layers weight & bias
params = {
    'h1': tf.get_variable("h1", shape=[n_features, n_hidden_1], dtype=dtype_tf, initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,  mode='FAN_IN', uniform=False,  seed=None, dtype=dtype_tf)),
    'h2': tf.get_variable("h2", shape=[n_hidden_1, n_hidden_2], dtype=dtype_tf, initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,  mode='FAN_IN', uniform=False,  seed=None, dtype=dtype_tf)),
    #'h3': tf.get_variable("h3", shape=[n_hidden_2, n_hidden_3], dtype=dtype_tf, initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,  mode='FAN_IN', uniform=False,  seed=None, dtype=dtype_tf)),
    #'h4': tf.get_variable("h4", shape=[n_hidden_3, n_hidden_4], dtype=dtype_tf, initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,  mode='FAN_IN', uniform=False,  seed=None, dtype=dtype_tf)),
    'hout': tf.get_variable("hout", shape=[n_hidden_2, n_classes], dtype=dtype_tf, initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,  mode='FAN_IN', uniform=False,  seed=None, dtype=dtype_tf)),
    'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1, dtype=dtype_tf), dtype=dtype_tf),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1, dtype=dtype_tf), dtype=dtype_tf),
    #'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1, dtype=dtype_tf), dtype=dtype_tf),
    #'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1, dtype=dtype_tf), dtype=dtype_tf),
    'bout': tf.Variable(tf.random_normal([n_classes], 0, 0.1, dtype=dtype_tf), dtype=dtype_tf)
}

model_vars = {
    'global_step' : tf.Variable(0, name='global_step', trainable=False, dtype=dtype_tf)
}

# Construct model
ypred = multilayer_perceptron(xbatch, params, train_pl)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(ypred-ybatch))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads = optimizer.compute_gradients(cost, var_list=list(params.values()))

#add noise
noisy_gradients = []
for grad in grads:
    denom = tf.pow( (1+model_vars["global_step"]), tf.cast(0.55,  dtype_tf))
    variance =  tf.cast(1/denom, dtype_tf)
    gradient_shape = grad[0].get_shape()
    noise = random_ops.truncated_normal(gradient_shape, stddev=tf.sqrt(variance), dtype=dtype_tf)
    noisy_gradients.append((grad[0] + noise, grad[1]))
grads = noisy_gradients

#norm grads
"""
gradients, variables = zip(*grads)
grads_normed, norms = tf.clip_by_global_norm(gradients, 10)
grads = list(zip(grads_normed, variables))
"""
train_step = optimizer.apply_gradients(grads, name="min_loss")
saver=tf.train.Saver(var_list=tf.trainable_variables())
# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(total_len/batch_size)
        # Loop over all batches
        """
        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([train_step, cost, ypred], feed_dict={xbatch: batch_x,
                                                          ybatch: batch_y, train_pl: True})
            # Compute average loss
            avg_cost += c / total_batch
        """
        #do mini batch 
        # Randomize data point
        X_train, Y_train = shuffle(X_train, Y_train)

        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([train_step, cost, ypred], feed_dict={xbatch: batch_x,
                                                          ybatch: batch_y, train_pl: True})
            # Compute average loss
            avg_cost += c / total_batch

        # sample prediction
        label_value = np.transpose(batch_y)
        estimate = p
        err = label_value-estimate
        print ("num batch:", total_batch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "Train cost=", \
                "{:.9f}".format(avg_cost))
            print ("[*]----------------------------")
            for i in range(3):
                ind = randint(0, batch_size-1)
                print ("label value:", label_value[ind], \
                    "estimated value:", estimate[ind])
            print ("[*]=================== Seed", seed)
        if epoch % test_cycle == 0:
            test_cost, yp_test = sess.run([cost, ypred], feed_dict={xbatch: X_test, ybatch: Y_test, train_pl: False})
            print ("Testing cost=", test_cost)
            print ("r2_score=", r2_score(np.transpose(Y_test), yp_test))
            saver.save(sess, './summaries_movies/' + str(seed) + '/model/',global_step=epoch)
    # Test model
    print ("Optimization Finished!")
    test_cost, yp_test = sess.run([cost, ypred], feed_dict={xbatch: X_test, ybatch: Y_test, train_pl: False})
    print ("Testing cost=", test_cost)
    print ("r2_score=", r2_score(np.transpose(Y_test), yp_test))
    saver.save(sess, './summaries_movies/' + str(seed) + '/model/',global_step=epoch)