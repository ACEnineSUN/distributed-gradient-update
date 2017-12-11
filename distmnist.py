#!/usr/bin/env python
# coding=utf-8

"""
This program test saving gradients, and subtract gradients 
When training CNN mnist classifying
One parameter server and two workers will be used to verify the program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def deepnn(x):
    """
    deepnn builds a graph for a deep net for classifying digits

    Args: 
    x: an input tensor with dimensions (N_examples, 784), where 784 is 
    the number of pixels in a standard mnist image

    Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with
    values equal to the logits of classifying the digit into one of 10 classes
    keep_prob is a scalar palceholder for the probability of dropout
    """

    #Reshape with a convolutional network
    #Last dimension is for features, since here is gray image, the value is 1
    #For RGB image, it will be 3
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    #First convolutional layer - maps one grayscale image to 32 feature maps
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    #Pooling layer-downsamples by 2x
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    #Second convolutional layer- mapes 32->64 feature maps
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    #Second pooling layer, downsamples by 2x
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    #Fully connected layer1- after two round downsampling, 28x28 images
    #now 7x7x64 feature maps -- maps this  to 1024 features
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #Dropout -- controls the complexity of the model
    #Prevents co-adaptation of features
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride"""
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    """max_poop_2x2 downsamples a feature map by 2x"""
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable for the given shape"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates bias variable for the given shape"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#define the cluster information
ps_hosts = ["localhost:2222"]
worker_hosts = ["localhost:2223", "localhost:2224"]
#worker_hosts = ["slave2:2223"]

cluster = tf.train.ClusterSpec({"ps":ps_hosts, "worker":worker_hosts})

#define the input flags
tf.app.flags.DEFINE_string("job_name","","one of ps or worker")
tf.app.flags.DEFINE_integer("task_index", 0, "task_index within the job")
FLAGS = tf.app.flags.FLAGS

#start a server for the specific task
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

#the hyper parameters for the network
#batch_size = 100
#learning_rate = 0.005
training_epochs = 60 
data_path = "./data"
logs_path = "./checkpoint"
#WORKER_NUMS = 1 

def main(_):
    #start training
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        #start between-graph replication
        with tf.device(tf.train.replica_device_setter(
            worker_device = "/job:worker/task:%d" % FLAGS.task_index,
            cluster = cluster)):
        
            #define the global step
            global_step = tf.get_variable('global_step', [], 
                                     initializer=tf.constant_initializer(0),
                                     trainable=False)
            #Define the model and input the data

            #Import data
            mnist = input_data.read_data_sets(data_path, one_hot = True)
            #task_index = FLAGS.task_index

            #Create model
            x = tf.placeholder(tf.float32, [None, 784])
            y_ = tf.placeholder(tf.float32, [None, 10])

            #Define the loss and build the graph
            y_conv, keep_prob = deepnn(x)

            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_,
                                                                   logits = y_conv)
            cross_entropy = tf.reduce_mean(cross_entropy)

            #The optimizer, to modify gradients later here
            with tf.name_scope('sgd_optimizer'):
                #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,global_step=global_step)
                opt = tf.train.GradientDescentOptimizer(0.01)
                grads_and_vars = opt.compute_gradients(cross_entropy)
                print("length of (g,v):",len(grads_and_vars))
                train_step = opt.apply_gradients(grads_and_vars, global_step=global_step)

            #The saver for the gradients
            with tf.name_scope('gradient_saver'):
                grads_save = [(tf.Variable(tf.zeros(g.get_shape()), trainable=False), v) for (g, v) in grads_and_vars]
                #grads_list = [ [(tf.Variable(tf.zeros(g.get_shape()), trainable=False), v) for (g, v) in grads_and_vars], 
                #              [(tf.Variable(tf.zeros(g.get_shape()), trainable=False), v) for (g, v) in grads_and_vars]]
                #zero_grads = [(tf.Variable(tf.zeros(g.get_shape()), trainable=False), v) for (g, v) in grads_and_vars]
                sub_save = [(tf.Variable(tf.zeros(g.get_shape()), trainable=False), v) for (g, v) in grads_and_vars]

            #The op for apply subtracted gradients for network
            with tf.name_scope('gradient_sub'):
                train_sub = opt.apply_gradients(sub_save, global_step=global_step)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
                correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

            #The gradients add ops
            #Attention here, grads_save is Variable, after assign_add, return a Tensor
            #So, here we must need a addops to maintain these tensors and can be outputed
            addops = []
            for i in range(len(grads_save)):
                addop = grads_save[i][0].assign_add(grads_and_vars[i][0])
                addops.append(addop)

            #The gradients subtract ops
            #Note here, grads_and_vars[i][0]->tensor, grads_save[i][0].value()->tensor 
            #subops list of tensors, so even after executed, this won't change grads_and_vars[i][0] 
            #define a variable maintain the subtracted value
            subops = []
            for i in range(len(grads_and_vars)):
                #subops.append(tempop)
                tempop = sub_save[i][0].assign_add(tf.subtract(grads_and_vars[i][0], grads_save[i][0].value()))
                subops.append(tempop)


            """
            #gradients check ops
            testops = []
            for i in range(len(grads_save)):
                boolop = tf.reduce_all(tf.equal(grads_save[i][0], grads_and_vars[i][0]))
                testops.append(boolop)
        
            #verify subtract ops
            versubops = []
            for i in range(len(grads_and_vars)):
                subboolop = tf.reduce_all(tf.equal(grads_and_vars[i][0], zero_grads[i][0]))
                versubops.append(subboolop)
            """
            
            init_op = tf.global_variables_initializer()
            
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            logdir = logs_path,
                            init_op = init_op,
                            global_step = global_step,
                            save_model_secs = 180)

        with sv.prepare_or_wait_for_session(server.target) as sess:
            for i in range(training_epochs):
                #print("global_step:", sess.run(global_step))
                batch = mnist.train.next_batch(50)
                
                train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1], keep_prob:1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
                sess.run(train_step, feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

                a = sess.run(global_step)
                if(a <50):
                    #Add the current gradients to the grads_save
                    print("global step:", a)
                    for j in range(len(addops)):
                        #print("grads_save[i]", sess.run(addops[i], feed_dict={x:batch[0], y_:batch[1], keep_prob:1}))
                        #print("grads_and_vars", sessrun(grads_and_vars[i], feed_dict={x:batch[0],y_:batch[1],keep_prob:1}))
                        sess.run(addops[j], feed_dict={x:batch[0], y_:batch[1], keep_prob:1})
                        #print(sess.run(grads_save[i][0].value(), feed_dict = {x:batch[0], y_:batch[1], keep_prob:1}))
                    #print(sess.run(grads_save[0][0].value(), feed_dict={x:batch[0], y_:batch[1], keep_prob:1}))
                    #print("current grads_and_vars:", grads_and_vars)
                    #print("grads_save:", grads_save)
                
                """
                #Test if grads_save equals grads_and_vars
                equaltest = True
                for i in range(len(testops)):
                    B
                    if not sess.run(testops[i], feed_dict={x:batch[0], y_:batch[1], keep_prob:1}):
                        equaltest = False
                print("equal test: ", equaltest)
                """
                if (a == 51):
                    print("global step:", a)
                    #Subtract the grads_and_vars with grads_save
                    for j in range(len(subops)):
                        sess.run(subops[j], feed_dict={x:batch[0], y_:batch[1], keep_prob:1})
                    #print(sess.run(sub_save[0][0].value(), feed_dict={x:batch[0], y_:batch[1], keep_prob:1}))
                    sess.run(train_sub, feed_dict={x:batch[0], y_:batch[1], keep_prob:1})
                    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1],keep_prob:1.0})
                    #print(sess.run(sub_save[0][0].value(), feed_dict={x:batch[0],y_:batch[1], keep_prob:1.0}))
                    print("step %d, training accuracy after subtract %g" % (i, train_accuracy))

                """
                #Test if grads_and_vars equal to zeros after subtract
                zerotest = True
                for i in range(len(versubops)):
                    if not sess.run(versubops[i], feed_dict={x:batch[0], y_:batch[1], keep_prob:1}):
                        print(i)
                        zerotest = False
                print("zero equal test: ", zerotest)
                """
                #for i in range(len(grads_and_vars)):
                #   print(sess.run(grads_and_vars[i][0]))


                #sess.run(train_step, feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

            print("Training Loop finish")
            #print("test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))


if __name__ == '__main__':
    tf.app.run()
