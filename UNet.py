# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:45:36 2017

@author: shengx
"""

import tensorflow as tf
import os

import Parameters as params

class UNet():
    def __init__(self):
        self.__IMG_X = params.IMG_X
        self.__IMG_Y = params.IMG_Y
        self.__IMG_Z = params.IMG_Z
        self.__IMG_OUT_Z = params.IMG_OUT_Z
        self.__L2_REG = params.WEIGHT_L2_REG
        
        self.global_step = tf.Variable(0, trainable=False)
        self.__starter_learning_rate = params.LEARNING_RATE
        self.learning_rate = tf.train.exponential_decay(learning_rate = self.__starter_learning_rate, 
                                                          global_step = self.global_step, 
                                                          decay_steps = params.LEARNING_RATE_DECAY_STEP, 
                                                          decay_rate = params.LEARNING_RATE_DECAY, 
                                                          staircase=True) 

        self.model = self.build()
        self.merged_summaries = tf.summary.merge_all()
       
    def build(self):
        image_in = tf.placeholder(tf.float32, shape=[None, self.__IMG_X, self.__IMG_Y, self.__IMG_Z]) # input image size 256 X 256 X 4
        label_in = tf.placeholder(tf.float32, shape=[None, self.__IMG_X, self.__IMG_Y, self.__IMG_OUT_Z])
        
        logits = self.u_net(image_in)
        image_out = tf.nn.sigmoid(logits)
        
        loss = self.loss_func(image_out, label_in)  
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        
        tf.summary.scalar('Loss', loss)
        
        return {'image_in': image_in, 'label_in':label_in,'loss':loss,
                'train_op':train_op, 'image_out': image_out}

    def u_net(self, image_in):
        layer_out = []
        filters_down = [64, 128, 256, 512, 1024]
        for layers in range(5):
            with tf.variable_scope('down_%d' %layers):
                if layers == 0:
                    tensor_out = self.down_block(image_in, filters_down[layers], False)
                else:
                    tensor_out = self.down_block(layer_out[-1], filters_down[layers])
                layer_out.append(tensor_out)
                    
        filters_up = [512, 256, 128, 64]
        for layers in range(4):
            with tf.variable_scope('up_%d' %layers):
                up_out = self.up_block(layer_out[3-layers], layer_out[-1], filters_up[layers])
                layer_out.append(up_out)
                
        filter = self.variables(name='final_conv', shape=[ 1, 1, filters_up[-1], self.__IMG_OUT_Z])
        image_out = tf.nn.conv2d(layer_out[-1], filter, strides=[1, 1, 1, 1], padding='SAME')    
        return image_out

    def down_block(self, tensor_in, channel_out, down_pool = True):
        channel_in = tensor_in.get_shape().as_list()[-1]
        if down_pool:
            tensor_in = tf.nn.max_pool(tensor_in, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        
        with tf.variable_scope('conv1'):
            filter_shape = [3, 3, channel_in, channel_out]
            filter = self.variables(name='conv', shape=filter_shape)
            conv1_out = tf.nn.conv2d(tensor_in, filter, strides=[1, 1, 1, 1], padding='SAME')
            conv1_out = tf.nn.elu(conv1_out)
        with tf.variable_scope('conv2'):
            filter_shape = [3, 3, channel_out, channel_out]
            filter = self.variables(name='conv', shape=filter_shape)
            conv2_out = tf.nn.conv2d(conv1_out, filter, strides=[1, 1, 1, 1], padding='SAME')
            conv2_out = tf.nn.elu(conv2_out)

        return conv2_out
    
    def up_block(self, shortcut_in, tensor_in, channel_out):
        channel_in = tensor_in.get_shape().as_list()[-1]
        
        with tf.variable_scope('deconv1'):
            filter_shape = [2, 2, channel_out, channel_in]
            filter = self.variables(name='deconv', shape=filter_shape)
            in_shape = tf.shape(tensor_in)
            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]//2])
            deconv1_out = tf.nn.conv2d_transpose(tensor_in, filter, out_shape, strides=[1, 2, 2, 1], padding='SAME')
            deconv1_out = tf.nn.elu(deconv1_out)
            
        with tf.variable_scope('concat'):
            concat_out = tf.concat([shortcut_in, deconv1_out], axis = 3)
            
        with tf.variable_scope('conv1'):
            filter_shape = [3, 3, 2 * channel_out, channel_out]
            filter = self.variables(name='conv', shape=filter_shape)
            conv1_out = tf.nn.conv2d(concat_out, filter, strides=[1, 1, 1, 1], padding='SAME')
            conv1_out = tf.nn.elu(conv1_out)            
        
        with tf.variable_scope('conv2'):
            filter_shape = [3, 3, channel_out, channel_out]
            filter = self.variables(name='conv', shape=filter_shape)
            conv2_out = tf.nn.conv2d(conv1_out, filter, strides=[1, 1, 1, 1], padding='SAME')
            conv2_out = tf.nn.elu(conv2_out) 
        
        return conv2_out
            
    def variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer()):
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.__L2_REG)
        variable =  tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)
        return variable
    
    def loss_func(self, prediction, truth):        
        loss = - truth * tf.log(prediction + 1e-6) - (1 - truth) * tf.log(1 - prediction + 1e-6)
        loss = tf.reduce_mean(loss)
        return loss
    
    def train(self, sess, image_in, label_in):
        _, loss, summ = sess.run([self.model['train_op'], self.model['loss'], self.merged_summaries], feed_dict = {
            self.model['image_in']:image_in,
            self.model['label_in']:label_in})
        return loss, summ

    def test(self, sess, image_in, label_in):
        loss, summ = sess.run([self.model['loss'], self.merged_summaries], feed_dict = {
            self.model['image_in']:image_in,
            self.model['label_in']:label_in})
        return loss, summ

    def predict(self, sess, image_in):
        image_out = sess.run(self.model['image_out'], feed_dict = {
            self.model['image_in']:image_in
            })       
        return image_out 
    
    
class Logger():
    # must be called within the graph after the nn construction
    def __init__(self):
        self.save_path = params.SAVE_PATH

        self.saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours = 3)

        self.train_writer = tf.summary.FileWriter(self.save_path +'train')
        self.test_writer = tf.summary.FileWriter(self.save_path +'test')
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path) 
            
    def restore(self, sess):
        
        try:
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(sess, load_path)        
            print('Network variables restored!')
        except:
            print('Cannot restore variables')
            
    def save(self, sess, epoches):        
        self.saver.save(sess, self.save_path+'model-'+str(epoches)+'.cptk')
        
    def save_summary(self, steps, train_summary = None, test_summary = None):
        if train_summary is not None:
            self.train_writer.add_summary(train_summary, steps)
        if test_summary is not None:
            self.test_writer.add_summary(test_summary, steps)
