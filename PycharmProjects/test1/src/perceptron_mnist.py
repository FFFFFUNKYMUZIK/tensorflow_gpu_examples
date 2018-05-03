# coding : utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

batch_size=100
train_size=10000

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

#input_data = np.float32(np.random.rand(train_size, 3))
#x=tf.placeholder(dtype=tf.float32, shape=[None, 3], name="input_data")
#W=tf.Variable([[3], [1.5], [2.5]], dtype=tf.float32)
#b=tf.Variable([[-5.44]], dtype=tf.float32)
#W_const=tf.constant([[-0.5], [-1], [1.5]], dtype=tf.float32)
#b_const=tf.constant([[-10]], dtype=tf.float32)
#y_label=tf.add(tf.matmul(x, W_const), b_const)
#y=tf.add(tf.matmul(x, W), b)

x=tf.placeholder(tf.float32, [None, 784])
W=tf.Variable(dtype=tf.float32, initial_value=tf.zeros([784, 10]))
b=tf.Variable(dtype=tf.float32, initial_value=tf.zeros([10]))

y=tf.matmul(x, W)+b

y_label=tf.placeholder(tf.float32, [None, 10])

# cost = 1/2*square(y-y_data)
#squared_deltas=tf.square(y-y_label)
#loss=1/2*tf.reduce_mean(squared_deltas)

# cost = cross entropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_label))
accuracy_correct=tf.equal(tf.argmax(y, axis=1), tf.argmax(y_label, axis=1))
accuracy = tf.reduce_mean(tf.cast(accuracy_correct, dtype=tf.float32))



train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


init_op = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init_op)

#print('[true value]')
#print('W : \n', sess.run(W_const))
#print('b : \n', sess.run(b_const))

iters_per_epoch=train_size/batch_size
epoch=0
max_epoch=100
max_iter=int(max_epoch*iters_per_epoch)
train_loss_list=[]
epoch_list=[]
accuracy_list=[]

# training
for step in range(0, max_iter):
    #batch_mask=np.random.choice(train_size, batch_size)
    #input_batch=input_data[batch_mask]
    batch_xs, batch_ys = mnist.train.next_batch(100)
    result=sess.run(train, feed_dict={x:batch_xs,y_label:batch_ys})

    if step % iters_per_epoch==0:

        epoch_list.append(epoch)
        print('step :', step, ' epoch :', epoch)
#        print('[trained]')
#        print('W : \n', sess.run(W))
#        print('b : \n', sess.run(b))
#        print('y-y_label : \n', sess.run(squared_deltas, feed_dict={x:input_data}))
        train_loss=sess.run(loss, feed_dict={x:batch_xs,y_label:batch_ys})
        print('loss : ', train_loss)
        train_loss_list.append(train_loss)
        epoch=epoch+1
        train_accuracy=sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels})
        accuracy_list.append(train_accuracy)
        print('accuracy :',train_accuracy)

fig1=plt.figure()
ax1handle = fig1.add_subplot(111)
ax1handle.plot(epoch_list, train_loss_list)
#plt.ylim(0, 0.01)
fig2=plt.figure()
ax2handle = fig2.add_subplot(111)
ax2handle.plot(epoch_list, accuracy_list)
plt.show()

# final accuracy test
print('final accuracy :', sess.run(accuracy, feed_dict={x:mnist.test.images, y_label: mnist.test.labels}))
