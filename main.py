# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import imageio.v2 as imageio

import os
from imageio import mimsave




from keras import layers
from keras.layers import Dense, Activation
import glob
#load images
training_data = glob.glob('data/*.jpg')
print(len(training_data))
#load tag
attributes = pd.read_csv('face_attributes.txt')
print(attributes)

attributes.index = attributes['#']
attributes.head()
print(training_data[0])

BatchSize = 64
alpha_factor = 0.04
beta = 3
z = 128

directory_out = 'result'
if not os.path.exists (directory_out):
    os.mkdir (directory_out)

width_image = 128
height_image = 128
label_image = 34

tf.compat.v1.disable_v2_behavior()
X = tf.compat.v1.placeholder (shape=[BatchSize, height_image, width_image, 3], dtype=tf.float32,name='X')
X_noise = tf.compat.v1.placeholder (shape=[BatchSize, height_image, width_image, 3],name='X_p',
                                    dtype=tf.float32)
Y = tf.compat.v1.placeholder (dtype=tf.float32, shape=[BatchSize, label_image],name='Y')
noise = tf.compat.v1.placeholder ( shape=[BatchSize, z],name='noise',dtype=tf.float32)
ynoise = tf.compat.v1.placeholder (dtype=tf.float32, shape=[BatchSize, label_image],
                                   name='noise_y')
bool_training = tf.compat.v1.placeholder (dtype=tf.bool, name='is_training')

steps = tf.Variable (0, trainable=False)
stepsadd = steps.assign_add (1)
initi_lrate = 0.0005
l_rate = tf.compat.v1.train.exponential_decay (initi_lrate,
                                               global_step=steps,
                                               decay_steps=20000,
                                               decay_rate=0.5)


def lrelu(x, leak=0.3):
    return tf.maximum (x, leak * x)


def sigmoid(logits, labels):
    return tf.nn.sigmoid_cross_entropy_with_logits (logits=logits, labels=labels)


def conv2d(inputs, kernel_size, filters, strides, padding='same',
           use_bias=True):
    return tf.keras.layers.Conv2D (kernel_size=kernel_size,
                             filters=filters, strides=strides,
                             padding=padding, use_bias=use_bias)(inputs)


def batch_norm(inputs, is_training=bool_training, d=0.75):


    return tf.keras.layers.BatchNormalization(momentum=d)(inputs)


#Discriminator

def D(previous, filters):
    x = 3
    j = 1
    hideenlayer\
        = lrelu (conv2d (previous, x, filters, j))
    hideenlayer = conv2d (hideenlayer, x, filters, j)
    hideenlayer = lrelu (tf.add (hideenlayer, previous))
    return hideenlayer


def discriminator(input, reuse=None):
    with tf.compat.v1.variable_scope ('D', reuse=reuse):
        h0 = input

        num = 32
        for i in range (5):
            if i < 3:
                h0 = lrelu (conv2d (h0, 4, num, 2))
            else:
                h0 = lrelu (conv2d (h0, 3, num, 2))
            h0 = D (h0, num)
            h0 = D (h0, num)
            num = 2*num

        h0 = lrelu (conv2d (h0, 3, num, 2))

        h0 =tf.keras.layers.Flatten()(h0)

        condition = tf.keras.layers.Dense (units=label_image) (h0)

        h0 = tf.keras.layers.Dense (units=1) (h0)
        return h0, condition

#Generator
def G(previous):
    x = 3
    j = 1
    k = 64
    hideenlayer = tf.nn.relu (
        batch_norm (conv2d (previous, x, k, j, use_bias=False)))
    hideenlayer = batch_norm (conv2d (hideenlayer, x, k, j, use_bias=False))
    hideenlayer = tf.add (hideenlayer, previous)
    return hideenlayer


def generator(z_dim, label):
    with tf.compat.v1.variable_scope ('G', reuse=None):

        dim = 16
        z_dim = tf.concat ([z_dim, label], axis=1)
        hiddenlayer = tf.keras.layers.Dense (units=dim * dim * 64) (z_dim)



        hiddenlayer = tf.reshape (hiddenlayer, shape=[-1, dim, dim, 64])
        hiddenlayer = tf.nn.relu (batch_norm (hiddenlayer))
        shortcut = hiddenlayer

        for i in range (16):
            hiddenlayer = G(hiddenlayer)

        hiddenlayer = tf.nn.relu (batch_norm (hiddenlayer))
        hiddenlayer = tf.add (hiddenlayer, shortcut)

        for i in range (3):
            hiddenlayer = conv2d (hiddenlayer, 3, 256, 1, use_bias=False)
            hiddenlayer = tf.compat.v1.depth_to_space (hiddenlayer, 2)
            hiddenlayer = tf.nn.relu (batch_norm (hiddenlayer))

        hiddenlayer = tf.keras.layers.Conv2D (kernel_size=9, filters=3, strides=1,
                               padding='same', activation=tf.nn.tanh,
                                use_bias=True)(hiddenlayer)
        return hiddenlayer

#loss function

g = generator(noise, ynoise)
real_d_data, realy = discriminator(X)
fake_d_data, fakey = discriminator(g, reuse=True)

d_real_loss = tf.reduce_mean(sigmoid(real_d_data, tf.ones_like(real_d_data)))
d_fake_loss = tf.reduce_mean(sigmoid(fake_d_data, tf.zeros_like(fake_d_data)))
g_fake_loss = tf.reduce_mean(sigmoid(fake_d_data, tf.ones_like(fake_d_data)))

lossc_r = tf.reduce_mean(sigmoid(realy, Y))
lossc_f = tf.reduce_mean(sigmoid(fakey, ynoise))

loss_d =  d_fake_loss + beta * lossc_r +d_real_loss
loss_g =  beta * lossc_f + g_fake_loss


alpha = tf.compat.v1.random_uniform(shape=[BatchSize, 1, 1, 1], minval=0., maxval=1.)
interpolates = alpha * X + (1 - alpha) * X_noise
grad = tf.gradients(discriminator(interpolates, reuse=True)[0], [interpolates])[0]
slop = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
gp = tf.reduce_mean((slop - 1.) ** 2)
loss_d += alpha_factor * gp


g_varis = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith('G')]

d_varis = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith('D')]

#optimizer


optimizor = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(optimizor):

    optimizer_d = tf.compat.v1.train.AdamOptimizer (learning_rate=l_rate,beta1=0.5).minimize (loss_d,var_list=d_varis)

    optimizer_g = tf.compat.v1.train.AdamOptimizer (learning_rate=l_rate,beta1=0.5).minimize (loss_g,var_list=g_varis)
#generate image
def generate_random_image(generated_imag):
    if isinstance(generated_imag, list):
        generated_imag = np.array(generated_imag)

    h = generated_imag.shape[1]
    w = generated_imag.shape[2]

    num = int(np.ceil(np.sqrt(generated_imag.shape[0])))
    condtion = 4
    if len(generated_imag.shape) == condtion and generated_imag.shape[3] == 3:
        imag = np.ones((num + generated_imag.shape[1] * num + 1,
                           num + generated_imag.shape[2] * num + 1, 3)) * 0.5
    elif len(generated_imag.shape) == condtion and generated_imag.shape[3] == 1:
        imag = np.ones((num + generated_imag.shape[1] * num + 1,
                           num + generated_imag.shape[2] * num + 1, 1)) * 0.5
    elif len(generated_imag.shape) == condtion - 1 :
        imag = np.ones((generated_imag.shape[1] * num + num + 1,
                           generated_imag.shape[2] * num + num + 1)) * 0.5


    for i in range(num):
        for j in range(num):
            this_filter = i * num + j
            if this_filter < generated_imag.shape[0]:
                this_img = generated_imag[this_filter]
                imag[ i + 1 + i * h:1 + i + h*(1 + i) ,
                         j + 1 + j * w:1 + j + w*(1 + j) ] = this_img

    return imag

import imageio.v2 as imageio


fullY = []
fullX = []

for i in tqdm (range (len (training_data))):

    data = imageio.imread (training_data[i])
    pts1 = np.float32(
        [[0, 0], [data.shape[1] - 1, 0], [0, data.shape[0] - 1], [data.shape[1] - 1, data.shape[0] - 1]])

    pts2 = np.float32([[0, 0], [127, 0], [0, 127], [127, 127]])


    M = cv2.getPerspectiveTransform(pts1, pts2)


    data = cv2.warpPerspective(data, M, (128, 128))
    data = (data / 255. - 0.5) * 2
    fullX.append (data)



    y = list(attributes.loc[training_data[i]])
    fullY.append (y[1:])

fullX = np.array (fullX)
fullY = np.array (fullY)
print (fullX.shape, fullY.shape)


def generateRandomAttributes():
    result = np.random.uniform(0.0, 1.0, [BatchSize, label_image]).astype(np.float32)
    result[result > 0.85] = 1
    result[result <= 0.85] = 0
    for i in range(BatchSize):
        haircolor = np.random.randint(0, 11)
        hs = np.random.randint(11, 15)
        eyecolor = np.random.randint(15, 28)
        result[i, :28] = 0
        result[i, haircolor] = 1
        result[i, hs] = 1
        result[i, eyecolor] = 1
    return result


sess = tf.compat.v1.Session ()
sess.run (tf.compat.v1.global_variables_initializer())
z_samples = np.random.uniform (-1.0, 1.0, [BatchSize, z]).astype (
    np.float32)
sampelsy = generateRandomAttributes ()
for i in range (BatchSize):
    sampelsy[i, :28] = 0
    sampelsy[i, i // 8 % 13] = 1
    sampelsy[i, i // 8 % 5 + 13] = 1
    sampelsy[i, i // 8 % 10 + 18] = 1
samples = []
loss = {'d': [], 'g': []}

default = 0
for i in tqdm (range (5000)):
    if default + BatchSize > fullX.shape[0]:
        default = 0
    if default == 0:
        data_index = np.arange (fullX.shape[0])
        np.random.shuffle (data_index)
        fullX = fullX[data_index, :, :, :]
        fullY = fullY[data_index, :]
    batch_X = fullX[default: default + BatchSize, :, :, :]
    batch_Y = fullY[default: default + BatchSize, :]
    perturb_X = batch_X + 0.5 * batch_X.std () * np.random.random (
        batch_X.shape)
    default += BatchSize

    n = np.random.uniform (-1.0, 1.0, [BatchSize, z]).astype (
        np.float32)
    ny = generateRandomAttributes ()
    u, d_ls = sess.run ([optimizer_d, loss_d], feed_dict={X: batch_X,
                                                          X_noise: perturb_X,
                                                          Y: batch_Y,
                                                          noise: n,
                                                          ynoise: ny,
                                                          bool_training: True})

    n = np.random.uniform (-1.0, 1.0, [BatchSize, z]).astype (
        np.float32)
    ny = generateRandomAttributes ()
    u, g_ls = sess.run ([optimizer_g, loss_g],
                        feed_dict={noise: n, ynoise: ny,
                                   bool_training: True})

    loss['d'].append (d_ls)
    loss['g'].append (g_ls)

    _, lr = sess.run ([stepsadd, l_rate])

    if i % 10 == 0:
        print (i, d_ls, g_ls, lr)
        fake_image = sess.run (g, feed_dict={noise: z_samples,
                                             ynoise: sampelsy,
                                             bool_training: False})
        fake_image = (fake_image + 1) / 2
        imgs = [img[:, :, :] for img in fake_image]
        fake_image = generate_random_image (imgs)
        plt.axis ('off')
        plt.imshow (fake_image)
        fake_image = (255 * fake_image).astype(np.uint8)
        imageio.imsave(os.path.join (directory_out, f'sample_{i}.jpg'),
                       fake_image)
        plt.show ()
        samples.append (fake_image)
plt.plot (loss['g'], label='generator')
plt.plot (loss['d'], label='discriminator')

plt.legend (loc='upper right')
plt.savefig ('loss function.png')
plt.show ()