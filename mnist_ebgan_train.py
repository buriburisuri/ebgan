# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32
z_dim = 50
margin = 10

#
# inputs
#

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=batch_size)

# input images
x = data.train.image

# generator labels ( all ones )
y = tf.ones(batch_size, dtype=tf.sg_floatx)

# discriminator labels ( half 1s, half 0s )
y_disc = tf.concat(0, [y, y * 0])

#
# create generator
#

# random uniform seed
z = tf.random_uniform((batch_size, z_dim))

with tf.sg_context(name='generator', size=4, stride=2, act='relu', bn=True):

    # generator network
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=7*7*128)
           .sg_reshape(shape=(-1, 7, 7, 128))
           .sg_upconv(dim=64)
           .sg_upconv(dim=1, act='sigmoid', bn=False))


#
# create discriminator
#

# create real + fake image input
xx = tf.concat(0, [x, gen])

with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu', bn=True):
    disc = (xx.sg_conv(dim=64)
            .sg_conv(dim=128)
            .sg_upconv(dim=64)
            .sg_upconv(dim=1, act='sigmoid', bn=False))


#
# loss & train ops
#

mse = tf.square(disc - xx)  # squared error
loss_disc = mse[:batch_size, :, :, :] + tf.maximum(margin - mse[batch_size:, :, :, :], 0)   # discriminator loss
loss_gen = mse[batch_size:, :, :, :]  # generator loss

train_disc = tf.sg_optim(loss_disc, lr=0.001, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_gen, lr=0.001, category='generator')  # generator train ops


#
# add summary
#

tf.sg_summary_loss(tf.identity(loss_disc, name='disc'))
tf.sg_summary_loss(tf.identity(loss_gen, name='gen'))
tf.sg_summary_image(gen)


#
# training
#

# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_disc, train_disc])[0]  # training discriminator
    l_gen = sess.run([loss_gen, train_gen])[0]  # training generator
    return np.mean(l_disc) + np.mean(l_gen)

# do training
alt_train(log_interval=10, ep_max=100, ep_size=data.train.num_batch, early_stop=False)

