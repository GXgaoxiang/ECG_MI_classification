# encoding: utf-8
import tensorflow as tf


def conv(layer_name, x, out_channels, kernel_size=None, stride=None, is_pretrain=True):
    """
    Convolution op wrapper, the Activation id ReLU
    :param layer_name: layer name, eg: conv1, conv2, ...
    :param x: input tensor, size = [batch_size, height, weight, channels]
    :param out_channels: number of output channel (convolution kernel)
    :param kernel_size: convolution kernel size, VGG use [3,3]
    :param stride: paper default = [1,1,1,1]
    :param is_pretrain: whether you need pre train, if you get parameter from other, you don not want to train again,
                        so trainable = false. if not trainable = true
    :return: 4D tensor
    """
    kernel_size = kernel_size if kernel_size else [3, 3]
    stride = stride if stride else [1, 1, 1, 1]

    in_channels = x.get_shape()[-1]

    with tf.variable_scope(layer_name):
        w = tf.get_variable(name="weights",
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=is_pretrain)
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0),
                            trainable=is_pretrain)
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')

        return x


def pool(layer_name, x, ksize=None, stride=None, is_max_pool=True):
    """
    Pooling op
    :param layer_name: layer name, eg:pool1, pool2,...
    :param x:input tensor
    :param ksize:pool kernel size, VGG paper use [1,2,2,1], the size of 2X2
    :param stride:stride size, VGG paper use [1,2,2,1]
    :param is_max_pool: default use max pool, if it is false, the we will use avg_pool
    :return: tensor
    """
    ksize = ksize if ksize else [1, 2, 2, 1]
    stride = stride if stride else [1, 2, 2, 1]

    if is_max_pool:
        x = tf.nn.max_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)

    return x


def batch_norm(x):
    """
    Batch Normalization (offset and scale is none). BN algorithm can improve train speed heavily.
    :param x: input tensor
    :return: norm tensor
    """
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)

    return x


def FC_layer(layer_name, x, out_nodes):
    """
    Wrapper for fully-connected layer with ReLU activation function
    :param layer_name: FC layer name, eg: 'FC1', 'FC2', ...
    :param x: input tensor
    :param out_nodes: number of neurons for FC layer
    :return: tensor
    """
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        # flatten into 1D
        flat_x = tf.reshape(x, [-1, size])

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)

        return x


def FC_layer_last(layer_name, x, out_nodes):
    """
    Wrapper for fully-connected layer with ReLU activation function
    :param layer_name: FC layer name, eg: 'FC1', 'FC2', ...
    :param x: input tensor
    :param out_nodes: number of neurons for FC layer
    :return: tensor
    """
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        # flatten into 1D
        flat_x = tf.reshape(x, [-1, size])

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        return x
############################################
def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network
from tflearn.layers.conv import global_avg_pool
def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not

from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm
def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :

        return batch_norm(x)

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

class_num = 2
def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')
training = tf.cast(True, tf.bool)
dropout_rate = 0.2
growth_k = 24
filters = growth_k
def bottleneck_layer(x, scope):
    with tf.name_scope(scope):
        x = Batch_Normalization(x, training=training, scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x, filter=4 * filters, kernel=[1,1], layer_name=scope+'_conv1')
        x = Drop_out(x, rate=dropout_rate, training=training)

        x = Batch_Normalization(x, training=training, scope=scope+'_batch2')
        x = Relu(x)
        x = conv_layer(x, filter=filters, kernel=[3,3], layer_name=scope+'_conv2')
        x = Drop_out(x, rate=dropout_rate, training=training)

        return x

def transition_layer(x, scope):
    with tf.name_scope(scope):
        x = Batch_Normalization(x, training=training, scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x, filter=filters, kernel=[1,1], layer_name=scope+'_conv1')
        x = Drop_out(x, rate=dropout_rate, training=training)
        x = Average_pooling(x, pool_size=[2,2], stride=2)

        return x

def dense_block(input_x, nb_layers, layer_name):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = Concatenation(layers_concat)
            x = bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
            layers_concat.append(x)

        x = Concatenation(layers_concat)

        return x
from tensorflow.contrib.layers import flatten
def Dense_net(input_x):
    # stride 2->1
    x = conv_layer(input_x, filter=2 * filters, kernel=[7,7], stride=1, layer_name='conv0')
    # x = Max_Pooling(x, pool_size=[3,3], stride=2)

    x = dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
    x = transition_layer(x, scope='trans_1')
    x = dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
    x = transition_layer(x, scope='trans_2')
    x = dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
    x = transition_layer(x, scope='trans_3')
    x = dense_block(input_x=x, nb_layers=32, layer_name='dense_final')
    # 100 Layer
    x = Batch_Normalization(x, training=training, scope='linear_batch')
    x = Relu(x)
    x = Global_Average_Pooling(x)
    x = flatten(x)
    x = Linear(x)

    return x

############################################

def Loss(logits, labels):
    """
    Compute loss
    :param logits: logits tensor, [batch_size, n_classes]
    :param labels: one_hot labels
    :return:
    """
    with tf.name_scope('loss') as scope:
        # use softmax_cross_entropy_with_logits(), so labels must be one-hot coding
        # if use sparse_softmax_cross_entropy_with_logits(), the labels not be one-hot
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss_temp = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope + '/loss', loss_temp)

        return loss_temp


def optimize(loss, learning_rate, global_step):
    """
    optimization, use Gradient Descent as default
    :param loss:
    :param learning_rate:
    :param global_step:
    :return:
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def Accuracy(logits, labels):
    """
    Evaluate quality of the logits at predicting labels
    :param logits: logits tensor, [batch_size, n_class]
    :param labels: labels tensor
    :return:
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.float32)
        print("correct:")
        print(correct)
        accuracy_temp = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + '/accuracy', accuracy_temp)

        return accuracy_temp

def Accuracy_TPR_TNR(predict, real):
    predictions = tf.argmax(predict, 1)
    actuals = tf.argmax(real, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )
    tpr = tf.div(tp_op,tf.add(tp_op,fn_op))
    fpr = tf.div(fp_op,tf.add(fp_op,tn_op))
    fnr = tf.div(fn_op,tf.add(tp_op,fn_op))
    tnr = tf.div(tn_op,tf.add(fp_op,tn_op))
    accuracy = tf.div(tf.add(tp_op,tn_op), tf.add(tp_op,tf.add(fp_op,tf.add(fn_op,tn_op))))

    recall = tpr
    precision = tf.div(tp_op,tf.add(tp_op,fp_op))

    f1_score = tf.div(2 * precision * recall,tf.add(precision,recall))
    return [accuracy,tpr,tnr,f1_score]

def num_correct_prediction(logits, labels):
    """
    Evaluate quality of the logits at predicting labels
    :param logits:
    :param labels:
    :return: number of correct prediction
    """
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)

    return n_correct


# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 22:55:47 2017
tensorflow : read my own dataset
@author: caokai
"""
import numpy as np
import tensorflow as tf


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename], shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           ####
                                           'img_raw1': tf.FixedLenFeature([], tf.string),
                                           'img_raw2': tf.FixedLenFeature([], tf.string),
                                           'img_raw3': tf.FixedLenFeature([], tf.string),
                                           'img_raw4': tf.FixedLenFeature([], tf.string),
                                           'img_raw5': tf.FixedLenFeature([], tf.string),
                                           'img_raw6': tf.FixedLenFeature([], tf.string),
                                           'img_raw7': tf.FixedLenFeature([], tf.string),
                                           'img_raw8': tf.FixedLenFeature([], tf.string),
                                           'img_raw9': tf.FixedLenFeature([], tf.string),
                                           'img_raw10': tf.FixedLenFeature([], tf.string),
                                           'img_raw11': tf.FixedLenFeature([], tf.string),
                                           'img_raw12': tf.FixedLenFeature([], tf.string),
                                           ####
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    ###
    img1 = tf.decode_raw(features['img_raw1'], tf.uint8)
    img2 = tf.decode_raw(features['img_raw2'], tf.uint8)
    img3 = tf.decode_raw(features['img_raw3'], tf.uint8)
    img4 = tf.decode_raw(features['img_raw4'], tf.uint8)
    img5 = tf.decode_raw(features['img_raw5'], tf.uint8)
    img6 = tf.decode_raw(features['img_raw6'], tf.uint8)
    img7 = tf.decode_raw(features['img_raw7'], tf.uint8)
    img8 = tf.decode_raw(features['img_raw8'], tf.uint8)
    img9 = tf.decode_raw(features['img_raw9'], tf.uint8)
    img10 = tf.decode_raw(features['img_raw10'], tf.uint8)
    img11 = tf.decode_raw(features['img_raw11'], tf.uint8)
    img12 = tf.decode_raw(features['img_raw12'], tf.uint8)
    ###
    img = tf.reshape(img, [128, 128, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    #####
    img1 = tf.reshape(img1, [128, 128, 3])
    img1 = tf.cast(img1, tf.float32) * (1. / 255) - 0.5
    img2 = tf.reshape(img2, [128, 128, 3])
    img2 = tf.cast(img2, tf.float32) * (1. / 255) - 0.5
    img3 = tf.reshape(img3, [128, 128, 3])
    img3 = tf.cast(img3, tf.float32) * (1. / 255) - 0.5
    img4 = tf.cast(img4, tf.float32) * (1. / 255) - 0.5
    img4 = tf.reshape(img4, [128, 128, 3])
    img5 = tf.cast(img5, tf.float32) * (1. / 255) - 0.5
    img5 = tf.reshape(img5, [128, 128, 3])
    img6 = tf.cast(img6, tf.float32) * (1. / 255) - 0.5
    img6 = tf.reshape(img6, [128, 128, 3])
    img7 = tf.cast(img7, tf.float32) * (1. / 255) - 0.5
    img7 = tf.reshape(img7, [128, 128, 3])
    img8 = tf.reshape(img8, [128, 128, 3])
    img8 = tf.cast(img8, tf.float32) * (1. / 255) - 0.5
    img9 = tf.reshape(img9, [128, 128, 3])
    img9 = tf.cast(img9, tf.float32) * (1. / 255) - 0.5
    img10 = tf.reshape(img10, [128, 128, 3])
    img10 = tf.cast(img10, tf.float32) * (1. / 255) - 0.5
    img11 = tf.reshape(img11, [128, 128, 3])
    img11 = tf.cast(img11, tf.float32) * (1. / 255) - 0.5
    img12 = tf.reshape(img12, [128, 128, 3])
    img12 = tf.cast(img12, tf.float32) * (1. / 255) - 0.5
    #####
    label = tf.cast(features['label'], tf.int32)

    print(label)
    return img, label, img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12


def distorted_input_train(filename, batch_size):
    img, label, img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12 = read_and_decode(filename)
    img_batch, label_batch, img1_batch, img2_batch, img3_batch, img4_batch, img5_batch, img6_batch, img7_batch, img8_batch, img9_batch, img10_batch, img11_batch, img12_batch = tf.train.shuffle_batch(
        [img, label, img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12],
        batch_size=batch_size, capacity=500,
        min_after_dequeue=100, num_threads=16)
    return img_batch, label_batch, img1_batch, img2_batch, img3_batch, img4_batch, img5_batch, img6_batch, img7_batch, img8_batch, img9_batch, img10_batch, img11_batch, img12_batch


def distorted_input_test(filename, batch_size):
    img, label, img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12 = read_and_decode(filename)
    img_batch, label_batch, img1_batch, img2_batch, img3_batch, img4_batch, img5_batch, img6_batch, img7_batch, img8_batch, img9_batch, img10_batch, img11_batch, img12_batch = tf.train.shuffle_batch(
        [img, label, img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12],
        batch_size=batch_size, capacity=500,
        min_after_dequeue=100, num_threads=16)
    return img_batch, label_batch, img1_batch, img2_batch, img3_batch, img4_batch, img5_batch, img6_batch, img7_batch, img8_batch, img9_batch, img10_batch, img11_batch, img12_batch


def one_hot(labels, Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
    return one_hot_label


def network(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, n_class, is_pretrain=True):
    with tf.name_scope("12brach"):
        with tf.name_scope("12branch_1"):
            x1 = conv('conv1_1', x1, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x1 = pool('pool1_1', x1, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x1 = conv('conv2_1', x1, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x1 = pool('pool2_1', x1, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        with tf.name_scope("12branch_2"):
            x2 = conv('conv1_2', x2, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x2 = pool('pool1_2', x2, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x2 = conv('conv2_2', x2, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x2 = pool('pool2_2', x2, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        with tf.name_scope("12branch_3"):
            x3 = conv('conv1_3', x3, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x3 = pool('pool1_3', x3, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x3 = conv('conv2_3', x3, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x3 = pool('pool2_3', x3, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        with tf.name_scope("12branch_4"):
            x4 = conv('conv1_4', x4, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x4 = pool('pool1_4', x4, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x4 = conv('conv2_4', x4, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x4 = pool('pool2_4', x4, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        with tf.name_scope("12branch_5"):
            x5 = conv('conv1_5', x5, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x5 = pool('pool1_5', x5, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x5 = conv('conv2_5', x5, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x5 = pool('pool2_5', x5, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        with tf.name_scope("12branch_6"):
            x6 = conv('conv1_6', x6, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x6 = pool('pool1_6', x6, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x6 = conv('conv2_6', x6, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x6 = pool('pool2_6', x6, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        with tf.name_scope("12branch_7"):
            x7 = conv('conv1_7', x7, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x7 = pool('pool1_7', x7, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x7 = conv('conv2_7', x7, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x7 = pool('pool2_7', x7, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        with tf.name_scope("12branch_8"):
            x8 = conv('conv1_8', x8, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x8 = pool('pool1_8', x8, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x8 = conv('conv2_8', x8, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x8 = pool('pool2_8', x8, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        with tf.name_scope("12branch_9"):
            x9 = conv('conv1_9', x9, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x9 = pool('pool1_9', x9, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x9 = conv('conv2_9', x9, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x9 = pool('pool2_9', x9, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        with tf.name_scope("12branch_10"):
            x10 = conv('conv1_10', x10, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x10 = pool('pool1_10', x10, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x10 = conv('conv2_10', x10, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x10 = pool('pool2_10', x10, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        with tf.name_scope("12branch_11"):
            x11 = conv('conv1_11', x11, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x11 = pool('pool1_11', x11, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x11 = conv('conv2_11', x11, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x11 = pool('pool2_11', x11, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        with tf.name_scope("12branch_12"):
            x12 = conv('conv1_12', x12, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x12 = pool('pool1_12', x12, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            x12 = conv('conv2_12', x12, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
            x12 = pool('pool2_12', x12, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
    with tf.name_scope("merge_part"):
        x_merge = tf.concat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], 3)
        x_logits = Dense_net(x_merge)
        return x_logits


epoch = 500
batch_size = 16
num_class = 2
learning_rate = 0.05
num_train = 820
num_test = 137


def train():
    # 读训练集
    img, label, img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12 = read_and_decode(
         "../input/MI_train3.tfrecords")
    img_batch, label_batch, img1_batch, img2_batch, img3_batch, img4_batch, img5_batch, img6_batch, img7_batch, img8_batch, img9_batch, img10_batch, img11_batch, img12_batch = tf.train.shuffle_batch(
        [img, label, img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12],
        batch_size=batch_size, capacity=500,
        min_after_dequeue=100, num_threads=64)
    img_test, label_test, img1_test, img2_test, img3_test, img4_test, img5_test, img6_test, img7_test, img8_test, img9_test, img10_test, img11_test, img12_test = read_and_decode(
       "../input/MI_test3.tfrecords")
    test_img_batch, test_label_batch, test_img1_batch, test_img2_batch, test_img3_batch, test_img4_batch, test_img5_batch, test_img6_batch, test_img7_batch, test_img8_batch, test_img9_batch, test_img10_batch, test_img11_batch, test_img12_batch = tf.train.shuffle_batch(
        [img_test, label_test, img1_test, img2_test, img3_test, img4_test, img5_test, img6_test, img7_test, img8_test,
         img9_test, img10_test, img11_test, img12_test],
        batch_size=batch_size, capacity=500,
        min_after_dequeue=100, num_threads=64)

    x = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x1 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x2 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x3 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x4 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x5 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x6 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x7 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x8 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x9 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x10 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x11 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    x12 = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
    y_ = tf.placeholder(tf.int32, [batch_size, 2])

    logits = network(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, num_class, True)
    loss = Loss(logits, y_)
    y_softmax = tf.nn.softmax(logits)
    accuracy = Accuracy(y_softmax, y_)
    Score = Accuracy_TPR_TNR(y_softmax,y_)
    my_global_step = tf.Variable(0, trainable=False, name='global_step')
    train_op = optimize(loss, learning_rate, my_global_step)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tf.global_variables_initializer().run()

        train_batch_idxs = int(num_train / batch_size)
        test_batch_idxs = int(num_test / batch_size)
        for i in range(epoch):
            train_loss, train_acc, train_Acc, train_tpr, train_tnr, train_f1, n_batch = 0, 0, 0, 0, 0, 0, 0
            test_loss, test_acc, test_Acc, test_tpr, test_tnr, test_f1, test_n_batch = 0, 0, 0, 0, 0, 0, 0
            for step in range(train_batch_idxs):
                Img_batch, Label_batch, Img1_batch, Img2_batch, Img3_batch, Img4_batch, Img5_batch, Img6_batch, Img7_batch, Img8_batch, Img9_batch, Img10_batch, Img11_batch, Img12_batch = sess.run(
                    [img_batch, label_batch, img1_batch, img2_batch, img3_batch, img4_batch, img5_batch, img6_batch,
                     img7_batch, img8_batch, img9_batch, img10_batch, img11_batch, img12_batch])
                Label_batch = one_hot(Label_batch, 2)
                _, err, ac , score_set = sess.run([train_op, loss, accuracy,Score],
                                      feed_dict={x: Img_batch, x1: Img1_batch, x2: Img2_batch,
                                                 x3: Img3_batch, x4: Img4_batch, x5: Img5_batch,
                                                 x6: Img6_batch, x7: Img7_batch, x8: Img8_batch,
                                                 x9: Img9_batch, x10: Img10_batch, x11: Img11_batch,
                                                 x12: Img12_batch, y_: Label_batch})

                train_loss += err
                train_acc += ac
                train_Acc += score_set[0]
                train_tpr += score_set[1]
                train_tnr += score_set[2]
                train_f1 += score_set[3]
                n_batch += 1
            if i % 10 == 0 and i != 0:
                train_loss = train_loss / n_batch
                train_acc = train_acc / n_batch
                train_Acc = (train_Acc / n_batch) * 100
                train_tpr = (train_tpr / n_batch) * 100
                train_tnr = (train_tnr / n_batch) * 100
                train_f1 = (train_f1 / n_batch) * 100
                print("Epoch: %d" % (i))
                print("***Trainset Loss: %.8f, Accuracy: %.8f" % (train_loss, train_acc))
                print("            Tpr: %.8f,  Tnr: %.8f" % (train_tpr,train_tnr))
                print("            Acc: %.8f,  f1_score: %.8f" % (train_Acc,train_f1))
                for step in range(test_batch_idxs):
                    test_Img_batch, test_Label_batch, test_Img1_batch, test_Img2_batch, test_Img3_batch, test_Img4_batch, test_Img5_batch, test_Img6_batch, test_Img7_batch, test_Img8_batch, test_Img9_batch, test_Img10_batch, test_Img11_batch, test_Img12_batch = sess.run(
                        [test_img_batch, test_label_batch, test_img1_batch, test_img2_batch, test_img3_batch,
                         test_img4_batch, test_img5_batch, test_img6_batch, test_img7_batch, test_img8_batch,
                         test_img9_batch, test_img10_batch, test_img11_batch, test_img12_batch])
                    test_Label_batch = one_hot(test_Label_batch, 2)
                    err_test, ac_test, score_set_test= sess.run([loss, accuracy,Score],
                                                        feed_dict={x: test_Img_batch, x1: test_Img1_batch,
                                                                   x2: test_Img2_batch,
                                                                   x3: test_Img3_batch, x4: test_Img4_batch,
                                                                   x5: test_Img5_batch,
                                                                   x6: test_Img6_batch, x7: test_Img7_batch,
                                                                   x8: test_Img8_batch,
                                                                   x9: test_Img9_batch, x10: test_Img10_batch,
                                                                   x11: test_Img11_batch,
                                                                   x12: test_Img12_batch, y_: test_Label_batch})

                    test_loss += err_test
                    test_acc += ac_test
                    test_Acc += score_set_test[0]
                    test_tpr += score_set_test[1]
                    test_tnr += score_set_test[2]
                    test_f1 += score_set_test[3]
                    test_n_batch += 1

                test_loss = test_loss / test_n_batch
                test_acc = test_acc / test_n_batch
                test_Acc = (test_Acc / test_n_batch) * 100
                test_tpr = (test_tpr / test_n_batch) * 100
                test_tnr = (test_tnr / test_n_batch) * 100
                test_f1 = (test_f1 / test_n_batch) * 100
                print("***Testset Loss: %.8f, Accuracy: %.8f" % (test_loss, test_acc))
                print("            Tpr: %.8f,  Tnr: %.8f" % (test_tpr, test_tnr))
                print("            Acc: %.8f,  f1_score: %.8f" % (test_Acc, test_f1))
        coord.request_stop()
        coord.join(threads)


train()