import tensorflow as tf
import numpy as np
import anchors

class FaceBox(object):
    def __init__(self, sess, input_shape, anchors, threshold):
        self.sess = sess
        self.input_shape = input_shape
        self.base_init = tf.truncated_normal_initializer(stddev=0.1) # Initialise weights
        self.reg_init = tf.contrib.layers.l2_regularizer(scale=0.1) # Initialise regularisation
        self.anchor_len = anchors[0]
        self.threshold = threshold
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

    def hard_negative_mining(self, conf_loss, pos_ids, num_pos, mult = 3):
        negatives = tf.logical_not(pos_ids)
        conf_loss_neg = tf.boolean_mask(conf_loss, negatives) # Extract negative confidence losses only
        conf_loss_pos = tf.boolean_mask(conf_loss, pos_ids)

        n_neg = tf.cast(mult * num_pos, tf.int32)
        n_neg = tf.minimum(n_neg, tf.shape(conf_loss_neg)[0]) # Cap maximum negative value to # negative boxes
        conf_loss_k_neg, _ = tf.nn.top_k(conf_loss_neg, k = n_neg, sorted  = True)

        return tf.concat((conf_loss_pos, conf_loss_k_neg), axis = 0)

    def compute_loss(self, loc_preds, conf_preds, loc_true, conf_true):
        positive_check = tf.cast(tf.greater(conf_true, 0), tf.float32)
        positive_count = tf.reduce_sum(positive_check)

        pos_ids = tf.reshape(tf.cast(positive_check, tf.bool), (tf.shape(conf_true)[0], self.anchor_len))

        # NOTE: this process collapses batches (each image can have a different # positive masks)
        pos_locs = tf.boolean_mask(loc_preds, pos_ids)
        pos_locs_targets = tf.boolean_mask(loc_true, pos_ids)
        pos_locs_loss = tf.losses.huber_loss(pos_locs_targets, pos_locs) # Smoothed L1 loss

        conf_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = conf_true, logits = conf_preds)
        conf_loss = self.hard_negative_mining(conf_loss, pos_ids, positive_count)

        # If no bounding boxes across all batches, return 0 loss (i.e. skip)
        loss = tf.cond(tf.equal(positive_count, 0), 
                lambda: tf.Variable(0, dtype = tf.float32), 
                lambda: (conf_loss + pos_locs_loss)/positive_count) # Return zero if no positive boxes
        return loss

    def build_graph(self):
        # Process inputs
        self.inputs =  tf.placeholder(tf.float32, shape = (None, self.input_shape[1], self.input_shape[2], self.input_shape[3]), name = "inputs")
        self.is_training = tf.placeholder(tf.bool, name = "is_training")
        DEBUG = True
        if DEBUG: print('Input shape: ', self.inputs.get_shape())
            
        bbox_locs = []
        bbox_confs = []

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
        bbox_locs.append(l)
        bbox_confs.append(c)

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
        bbox_locs.append(l)
        bbox_confs.append(c)

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
        bbox_locs.append(l)
        bbox_confs.append(c)

        self.out_locs = tf.concat([tf.reshape(i, [tf.shape(i)[0], -1, 4]) for i in bbox_locs], axis = -2)
        self.out_confs = tf.concat([tf.reshape(i, [tf.shape(i)[0], -1, 2]) for i in bbox_confs], axis = -2)
        self.out_locs = tf.reshape(self.out_locs, [tf.shape(self.out_locs)[0], self.anchor_len, 4])
        self.out_confs = tf.reshape(self.out_confs, [tf.shape(self.out_confs)[0], self.anchor_len, 2])
        self.p_confs = tf.nn.softmax(self.out_confs)

        print('Output loc shapes' , self.out_locs.get_shape())
        print('Output conf shapes' , self.out_confs.get_shape())

        self.target_locs = tf.placeholder(tf.float32, shape = (None, self.anchor_len, 4), name = 'target_locs')
        self.target_confs = tf.placeholder(tf.float32, shape = (None, self.anchor_len, 1), name = 'target_confs')
        
        self.loss = self.compute_loss(self.out_locs, self.out_confs, self.target_locs, self.target_confs)
        self.mean_loss = tf.reduce_mean(self.loss)
        tf.summary.scalar('Loss', self.mean_loss)
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.extra_update_ops):
            self.train = tf.train.AdamOptimizer(0.01).minimize(self.mean_loss)
        self.merged = tf.summary.merge_all()

    def train_iter(self, anchors_vec, imgs, lbls):
        locs, confs = anchors.encode_batch(anchors_vec, lbls, threshold = 0.25)
        feed_dict = {
            self.inputs: imgs,
            self.is_training: True,
            self.target_locs: locs,
            self.target_confs: confs
        }
        pred_confs, pred_locs, summary, _, loss = self.sess.run([self.p_confs, self.out_locs, self.merged, self.train, self.mean_loss], feed_dict = feed_dict)
        return pred_confs, pred_locs, loss, summary
    
    def test_iter(self, imgs):
        feed_dict = {
            self.inputs: imgs,
            self.is_training: False
        }
        pred_confs, pred_locs = self.sess.run(self.p_confs, self.out_locs, feed_dict = feed_dict)
        return pred_confs, pred_locs
