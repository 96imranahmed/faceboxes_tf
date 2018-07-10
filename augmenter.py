import tensorflow as tf 
import numpy as np
import math as m

class AugmenterGPU(object):
    def __init__(self, sess, img_size):
        self.sess = sess
        self.size_out = img_size
        self.build_augment()
    
    def random_rotate(self, img, boxes):
        with tf.variable_scope('rotate'):
            r_a = tf.random_uniform(shape = (), minval = 0.0, maxval = tf.constant(m.pi), dtype = tf.float32)
            img = tf.contrib.image.rotate(img, r_a, interpolation = "NEAREST")
            centers = (boxes[:, :2] + boxes[:, 2:])/2
            w_h = boxes[:, 2:] - boxes[:, :2]
            cos_v, sin_v = tf.cos(r_a), tf.sin(r_a)

            w_h_n = tf.transpose(tf.matmul([[cos_v, sin_v], [sin_v, cos_v]], tf.transpose(w_h)))
            ctr = [[self.size_out[0]/2, self.size_out[1]/2]]
            centers -= tf.tile(ctr, [tf.shape(boxes)[0], 1])
            r_mat = [[cos_v, sin_v], [-sin_v, cos_v]]

            centers_n = tf.transpose(tf.matmul(r_mat, tf.transpose(centers)))
            centers_n += tf.tile(ctr, [tf.shape(boxes)[0], 1])
            boxes_n = tf.concat([tf.clip_by_value(centers_n - w_h_n/2, 0, self.size_out[0]), tf.clip_by_value(centers_n + w_h_n/2,0, self.size_out[1])], axis = 1) # Clip by value
            return img, boxes_n, r_a

    def build_augment(self):
        with tf.variable_scope('augment'):
            self.image_in = tf.placeholder(tf.float32, (self.size_out[0], self.size_out[1], 3))
            self.boxes_in = tf.placeholder(tf.float32, (None, 4))
            img, boxes = self.image_in, self.boxes_in

            img, boxes, r_ang = self.random_rotate(img, boxes)
            
            self.params = {'ang': r_ang}
            self.image_out = img
            self.boxes_out = boxes

    def augment_batch(self, imgs, lbls):
        imgs_out = []
        boxes_out = []
        aug_out = []
        for i in range(imgs.shape[0]):
            out_arr = {}
            feed_dict = {
                self.image_in: np.squeeze(imgs[i]), 
                self.boxes_in: lbls[i]
            }
            img, boxes, params = self.sess.run([self.image_out,
                                               self.boxes_out, 
                                               self.params], feed_dict = feed_dict)
            imgs_out.append(img)
            boxes_out.append(np.array(boxes, dtype = np.uint32)) # Can be 0-1024
            aug_out.append(params)
        return np.array(imgs_out, dtype = np.uint8), boxes_out, aug_out
