import tensorflow as tf
import numpy as np

class FaceBox(object):
    def __init__(self, sess, input_shape, anchors):
        self.sess = sess
        self.input_shape = input_shape
        self.base_init = tf.truncated_normal_initializer(stddev=0.1) # Initialise weights
        self.reg_init = tf.contrib.layers.l2_regularizer(scale=0.1) # Initialise regularisation
        self.build_graph()
    
    def CReLU(self, in_x, name):
        x = tf.layers.batch_normalization(in_x, training = self.is_training, name = name + '_batch')
        return tf.nn.crelu(x, name = name + '_crelu')

    def Inception(self, in_x, name):
        DEBUG = False
        if DEBUG: print('Input shape: ', in_x.get_shape())
        path_1 = tf.layers.conv2d(in_x, 32, 
                                kernel_size = [1, 1],
                                strides = 1,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = name + 'conv_1_1',
                                padding = 'SAME')
        path_2 = tf.layers.max_pooling2d(in_x, [3,3], 1, name = name+'pool_1_2',
                                padding = 'SAME') # No striding to preserve shape
        path_2 = tf.layers.conv2d(path_2, 32, 
                                kernel_size = [1, 1],
                                strides = 1,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = name + 'conv_1_2',
                                padding = 'SAME')
        if DEBUG: print('Path 2 shape: ', path_2.get_shape())
        path_3 = tf.layers.conv2d(in_x, 24, 
                                kernel_size = [1, 1],
                                strides = 1,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = name + 'conv_1_3',
                                padding = 'SAME')
        path_3 = tf.layers.conv2d(path_3, 32, 
                                kernel_size = [3, 3],
                                strides = 1,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = name + 'conv_2_3',
                                padding = 'SAME')
        if DEBUG: print('Path 3 shape: ', path_3.get_shape())
        path_4 = tf.layers.conv2d(in_x, 24, 
                        kernel_size = [1, 1],
                        strides = 1,
                        kernel_initializer=self.base_init,
                        kernel_regularizer=self.reg_init,
                        name = name + 'conv_1_4',
                        padding = 'SAME')
        path_4 = tf.layers.conv2d(path_4, 32, 
                        kernel_size = [3, 3],
                        strides = 1,
                        kernel_initializer=self.base_init,
                        kernel_regularizer=self.reg_init,
                        name = name + 'conv_2_4',
                        padding = 'SAME')
        path_4 = tf.layers.conv2d(in_x, 32, 
                        kernel_size = [3, 3],
                        strides = 1,
                        kernel_initializer=self.base_init,
                        kernel_regularizer=self.reg_init,
                        name = name + 'conv_3_4',
                        padding = 'SAME')
        if DEBUG: print('Path 4 shape: ', path_4.get_shape())
        return tf.concat([path_1, path_2, path_3, path_4], axis = -1)

    def build_graph(self):
        # Process inputs
        self.inputs =  tf.placeholder(tf.float32, shape = self.input_shape, name = "inputs")
        self.is_training = tf.placeholder(tf.bool, name = "is_training")

        # Rapidly Digested Convolutional Layers
        print('Building RDCL...')
        conv_1 = tf.layers.conv2d(self.inputs, 24, 
                                kernel_size = [7, 7],
                                strides = 4,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = 'Conv1',
                                padding = 'SAME')
        conv_1_crelu = self.CReLU(conv_1, 'CReLU_1')
        conv_1_pool = tf.layers.max_pooling2d(conv_1_crelu, [3,3],2, name = 'Pool1')
        conv_2 = tf.layers.conv2d(conv_1_pool, 64, 
                                kernel_size = [5, 5],
                                strides = 2,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = 'Conv2',
                                padding = 'SAME')
        conv_2_crelu = self.CReLU(conv_2, 'CReLU_2')
        conv_2_pool = tf.layers.max_pooling2d(conv_2_crelu, [3,3], 2, name = 'Pool2')

        print('Building Inception...')
        incept_1 = self.Inception(conv_2_pool, 'inception_1')
        incept_2 = self.Inception(incept_1, 'inception_2')
        incept_3 = self.Inception(incept_2, 'inception_3')

        print('Building MSCL...')
        conv_3_1 = tf.layers.conv2d(incept_3, 128, 
                                kernel_size = [1, 1],
                                strides = 1,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = 'Conv3_1',
                                padding = 'SAME')
        conv_3_2 = tf.layers.conv2d(conv_3_1, 256, 
                                kernel_size = [3, 3],
                                strides = 2,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = 'Conv3_2',
                                padding = 'SAME')
        conv_4_1 = tf.layers.conv2d(conv_3_2, 128, 
                                kernel_size = [1, 1],
                                strides = 1,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = 'Conv4_1',
                                padding = 'SAME')
        conv_4_2 = tf.layers.conv2d(conv_4_1, 256, 
                                kernel_size = [3, 3],
                                strides = 2,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = 'Conv4_2',
                                padding = 'SAME')