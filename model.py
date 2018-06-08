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

    def build_anchor(self, in_x, num_out, name):
        DEBUG = True
        bbox_loc_conv = tf.layers.conv2d(in_x, num_out*4, 
                        kernel_size = [3, 3],
                        strides = 1,
                        kernel_initializer=self.base_init,
                        kernel_regularizer=self.reg_init,
                        name = name + '_anchor_loc_conv',
                        padding = 'SAME')
        bbox_class_conv = tf.layers.conv2d(in_x, num_out*2, 
                        kernel_size = [3, 3],
                        strides = 1,
                        kernel_initializer=self.base_init,
                        kernel_regularizer=self.reg_init,
                        name = name + '_anchor_conf_conv',
                        padding = 'SAME')
        if DEBUG: print(name, 'anchor class shape: ', bbox_class_conv.get_shape()
            , ' anchor loc shape: ', bbox_loc_conv.get_shape())
        return bbox_loc_conv, bbox_class_conv

    def build_graph(self):
        # Process inputs
        self.inputs =  tf.placeholder(tf.float32, shape = (None, self.input_shape[1], self.input_shape[2], self.input_shape[3]), name = "inputs")
        self.is_training = tf.placeholder(tf.bool, name = "is_training")
        DEBUG = True
        if DEBUG: print('Input shape: ', self.inputs.get_shape())
            
        self.bbox_locs = []
        self.bbox_confs = []

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
        conv_1_pool = tf.layers.max_pooling2d(conv_1_crelu, [3,3],2, name = 'Pool1',  padding = 'SAME')
        if DEBUG: print('Conv 1 shape: ', conv_1_pool.get_shape())        
        conv_2 = tf.layers.conv2d(conv_1_pool, 64, 
                                kernel_size = [5, 5],
                                strides = 2,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = 'Conv2',
                                padding = 'SAME')
        conv_2_crelu = self.CReLU(conv_2, 'CReLU_2')
        conv_2_pool = tf.layers.max_pooling2d(conv_2_crelu, [3,3], 2, name = 'Pool2',  padding = 'SAME')
        if DEBUG: print('Conv 2 shape: ', conv_2_pool.get_shape())

        print('Building Inception...')
        incept_1 = self.Inception(conv_2_pool, 'inception_1')
        if DEBUG: print('Incept 1 shape: ', incept_1.get_shape())   
        incept_2 = self.Inception(incept_1, 'inception_2')
        if DEBUG: print('Incept 2 shape: ', incept_2.get_shape())   
        incept_3 = self.Inception(incept_2, 'inception_3')
        if DEBUG: print('Incept 3 shape: ', incept_3.get_shape())   
        
        if DEBUG: print('Inception 3 anchors...')
        l, c = self.build_anchor(incept_3, 21, 'incept_3')
        self.bbox_locs.append(l)
        self.bbox_confs.append(c)

        print('Building MSCL...')
        conv_3_1 = tf.layers.conv2d(incept_3, 128, 
                                kernel_size = [1, 1],
                                strides = 1,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = 'Conv3_1',
                                padding = 'SAME')
        if DEBUG: print('Conv 3_1 shape: ', conv_3_1.get_shape())   
        conv_3_2 = tf.layers.conv2d(conv_3_1, 256, 
                                kernel_size = [3, 3],
                                strides = 2,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = 'Conv3_2',
                                padding = 'SAME')

        if DEBUG: print('Conv 3_2 anchors...')
        l, c = self.build_anchor(conv_3_2, 1, 'conv_3_2')
        self.bbox_locs.append(l)
        self.bbox_confs.append(c)

        if DEBUG: print('Conv 3_2 shape: ', conv_3_2.get_shape())
        conv_4_1 = tf.layers.conv2d(conv_3_2, 128, 
                                kernel_size = [1, 1],
                                strides = 1,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = 'Conv4_1',
                                padding = 'SAME')
        if DEBUG: print('Conv 4_1 shape: ', conv_4_1.get_shape())
        conv_4_2 = tf.layers.conv2d(conv_4_1, 256, 
                                kernel_size = [3, 3],
                                strides = 2,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                name = 'Conv4_2',
                                padding = 'SAME')
        if DEBUG: print('Conv 4_2 shape: ', conv_4_2.get_shape())

        if DEBUG: print('Conv 4_2 anchors...')
        l, c = self.build_anchor(conv_4_2, 1, 'conv_4_2')
        self.bbox_locs.append(l)
        self.bbox_confs.append(c)
        
        out_locs = tf.concat([tf.reshape(i, [tf.shape(i)[0], -1, 4]) for i in self.bbox_locs], axis = -2)
        out_confs = tf.concat([tf.reshape(i, [tf.shape(i)[0], -1, 2]) for i in self.bbox_confs], axis = -2)

        print('Output loc shapes' , out_locs.get_shape())
        print('Output conf shapes' , out_confs.get_shape())

