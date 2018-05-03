# coding : utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

batch_size=100
train_size=10000


mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

num_filters1=32

x=tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,num_filters1], stddev=0.1))
h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME')
b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
h_conv1_cutoff = tf.nn.relu(h_conv1+b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

num_filters2=64

W_conv2 = tf.Variable(tf.truncated_normal([5,5,num_filters1,num_filters2], stddev=0.1))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME')
b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
h_conv2_cutoff = tf.nn.relu(h_conv2+b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filters2])

num_units1 = 7*7*num_filters2
num_units2 = 1024

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2)+b2)

keep_prob = tf.placeholder(tf.float32)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w0 = tf.Variable(tf.zeros([num_units2, 10]))
b0 = tf.Variable(tf.zeros([10]))
k = tf.matmul(hidden2_drop, w0) + b0
p = tf.nn.softmax(k)


t=tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=k,labels=t))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

saver.restore(sess, './cnn_mnist/cnn_mnist_session')

fig1 = plt.figure(figsize=(15,5))

subplot1=fig1.add_subplot(1,3,1)

subplot2 = fig1.add_subplot(1, 3, 2)

subplot3=fig1.add_subplot(1,3,3)

plt.show(False)
for i in range (100):
    subplot1.clear()
    subplot2.clear()
    subplot3.clear()

    subplot1.set_xticks(range(10))
    subplot1.set_xlim(-0.5, 9.5)
    subplot1.set_ylim(0, 1)

    subplot2.set_xticks(range(10))
    subplot2.set_xlim(-0.5, 9.5)
    subplot2.set_ylim(0, 1)

    subplot3.set_xticks([])
    subplot3.set_yticks([])


    x_test, t_test = mnist.test.next_batch(1)
    p_val = sess.run(p, feed_dict={x:x_test , t:t_test, keep_prob:1})
    print(p_val)

    p1=p_val[0]
    subplot1.bar(range(10), p1, align='center')

    subplot2.bar(range(10), t_test[0], align='center')

    subplot3.imshow(sess.run(tf.reshape(x_test, [28, 28])), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show(False)
