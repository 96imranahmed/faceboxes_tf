import tensorflow as tf 
import numpy as np
import cv2
import math as m
from random import randint

class AugmenterGPU(object):
    def __init__(self, sess, img_size):
        self.sess = sess
        self.size_out = img_size
        self.build_augment()
        self.build_random_crop()
    
    def random_rotate(self, img, boxes):
        with tf.name_scope('rotate'):
            r_a = tf.random_uniform(shape = (), minval = 0.0, maxval = 2*tf.constant(m.pi), dtype = tf.float32)
            img = tf.contrib.image.rotate(img, r_a, interpolation = "NEAREST")
            centers = (boxes[:, :2] + boxes[:, 2:])/2
            w_h = boxes[:, 2:] - boxes[:, :2]
            cos_v, sin_v = tf.cos(r_a), tf.sin(r_a)

            w_h_n = tf.transpose(tf.matmul(tf.abs([[(cos_v), sin_v], [sin_v, cos_v]]), tf.transpose(w_h)))
            ctr = [[self.size_out[0]/2, self.size_out[1]/2]]
            centers -= tf.tile(ctr, [tf.shape(boxes)[0], 1])
            r_mat = [[cos_v, sin_v], [-sin_v, cos_v]]

            centers_n = tf.transpose(tf.matmul(r_mat, tf.transpose(centers)))
            centers_n += tf.tile(ctr, [tf.shape(boxes)[0], 1])
            boxes_n = tf.concat([tf.clip_by_value(centers_n - w_h_n/2.0, 0, self.size_out[0]), tf.clip_by_value(centers_n + w_h_n/2,0, self.size_out[1])], axis = 1) # Clip by value
            return img, boxes_n, r_a
    
    def random_flip_lr(self, img, boxes, p = 0.5):
        def flip(img, boxes):
            fl_img = tf.image.flip_left_right(img)
            xmin, ymin, xmax, ymax = tf.unstack(boxes, axis=1)
            flipped_xmin = tf.subtract(float(self.size_out[0]), xmax)
            flipped_xmax = tf.subtract(float(self.size_out[0]), xmin)
            flipped_boxes = tf.stack([flipped_xmin, ymin, flipped_xmax, ymax], 1)
            return fl_img, flipped_boxes, True
        
        with tf.name_scope('flip_lr'):
            return tf.cond(tf.less(tf.random_uniform(shape = ()), p),
                           lambda: flip(img, boxes),
                           lambda: (img, boxes, False))

    def random_flip_ud(self, img, boxes, p = 0.5):
        def flip(img, boxes):
            fl_img = tf.image.flip_up_down(img)
            xmin, ymin, xmax, ymax = tf.unstack(boxes, axis=1)
            flipped_ymin = tf.subtract(float(self.size_out[0]), ymax)
            flipped_ymax = tf.subtract(float(self.size_out[0]), ymin)
            flipped_boxes = tf.stack([xmin, flipped_ymin, xmax, flipped_ymax], 1)
            return fl_img, flipped_boxes, True
        
        with tf.name_scope('flip_ud'):
            return tf.cond(tf.less(tf.random_uniform(shape = ()), p),
                           lambda: flip(img, boxes),
                           lambda: (img, boxes, False))

    def random_color_mutation(self, image): 
        P_RANDOM_BRIGHTNESS = 0.3
        BRIGHTNESS_AUG_PARAM = 0.2
        P_RANDOM_CONTRAST = 0.3
        CONTRAST_AUG_PARAM_1 = 0.8
        CONTRAST_AUG_PARAM_2 = 1.2
        P_RANDOM_HUE = 0.3
        HUG_AUG_PARAM = 0.2
        P_RANDOM_SAT = 0.3
        SAT_AUG_PARAM_1 = 0.8
        SAT_AUG_PARAM_2 = 1.2
        P_GRAYSCALE = 0.2
        P_PXL_SCALE = 0.7
        P_S_AND_P = 0.8

        with tf.name_scope('color_augment'):
            colour_augs = {}

            image, did_aug = tf.cond(tf.less(tf.random_uniform(shape = ()), P_RANDOM_BRIGHTNESS),
                            lambda: (tf.image.random_brightness(image, BRIGHTNESS_AUG_PARAM), True),
                            lambda: (image, False))
            colour_augs['brightness'] = did_aug
            image, did_aug = tf.cond(tf.less(tf.random_uniform(shape = ()), P_RANDOM_CONTRAST),
                            lambda: (tf.image.random_contrast(image, CONTRAST_AUG_PARAM_1, CONTRAST_AUG_PARAM_2), True),
                            lambda: (image, False))
            colour_augs['contrast'] = did_aug
            image, did_aug = tf.cond(tf.less(tf.random_uniform(shape = ()), P_RANDOM_HUE),
                            lambda: (tf.image.random_hue(image, HUG_AUG_PARAM), True),
                            lambda: (image, False))
            colour_augs['hue'] = did_aug
            image, did_aug = tf.cond(tf.less(tf.random_uniform(shape = ()), P_RANDOM_SAT),
                            lambda: (tf.image.random_saturation(image, SAT_AUG_PARAM_1, SAT_AUG_PARAM_2), True),
                            lambda: (image, False))
            colour_augs['saturation'] = did_aug

            def to_grayscale(image):
                image = tf.image.rgb_to_grayscale(image)
                image = tf.image.grayscale_to_rgb(image)
                return image, True

            def dropout_salt_and_pepper(image):
                P_DROPOUT_TOL = 0.05
                P_CHOOSE_RGB_VS_R_G_B = 0.7
                P_CHOOSE_HIGH_SINGLE = 0.5
                P_CHOOSE_SINGLE_MIXED = 0.7
                d_p = tf.random_uniform(shape = (), minval = 0, maxval = P_DROPOUT_TOL)

                def separate_switch_and_pepper(image, d_p):
                    P_PICK_HIGH = 0.5
                    switch = tf.less(tf.random_uniform(shape = tf.shape(image)),d_p/3)
                    s_p = 255.0 * tf.cast(tf.less(tf.random_uniform(shape = tf.shape(switch)), P_PICK_HIGH), tf.float32)*tf.cast(switch, tf.float32)
                    return switch, s_p

                def conc_switch_and_pepper(image, d_p):
                    P_PICK_HIGH = 0.5
                    switch = tf.less(tf.random_uniform(shape = (tf.shape(image)[0], tf.shape(image)[1], 1)), d_p)
                    s_p = 255.0 * tf.cast(tf.less(tf.random_uniform(shape = tf.shape(switch)), P_PICK_HIGH), tf.float32)*tf.cast(switch, tf.float32)
                    switch = tf.tile(switch, (1,1,3))
                    s_p = tf.tile(s_p, (1,1,3))
                    return switch, s_p

                switch, s_and_p = tf.cond(
                    tf.less(tf.random_uniform(shape = ()), P_CHOOSE_RGB_VS_R_G_B), # Random choose whether to set entire RGB (True) or R, G, B separately
                    lambda: conc_switch_and_pepper(image, d_p),
                    lambda: separate_switch_and_pepper(image, d_p)
                )

                one_or_zero = tf.cond(
                    tf.less(tf.random_uniform(shape = ()), P_CHOOSE_HIGH_SINGLE),
                    lambda: 255.0 * tf.ones_like(image),
                    lambda: 0.0 * tf.ones_like(image)
                ) # Random choose whether to set colour as consistently high or low 

                one_or_zero = tf.cond(
                    tf.less(tf.random_uniform(shape = ()), P_CHOOSE_SINGLE_MIXED),
                    lambda: one_or_zero,
                    lambda: s_and_p
                ) # Random choose whether to set salt and pepper (mixed) or fixed coloring

                image = tf.where(switch, one_or_zero, image)
                return image, True
            
            def random_value_scale(image):
                SCALE_PXL_MIN = 0.8
                SCALE_PXL_MAX = 1.2
                scale_size = tf.random_uniform(
                    tf.shape(image), minval=SCALE_PXL_MIN,
                    maxval=SCALE_PXL_MAX, dtype=tf.float32
                )
                image = tf.multiply(image, scale_size)
                image = tf.clip_by_value(image, 0.0, 255.0)
                return image, True

            image, did_aug = tf.cond(tf.less(tf.random_uniform(shape = ()), P_GRAYSCALE),
                    lambda: to_grayscale(image),
                    lambda: (image, False))

            colour_augs['grayscale'] = did_aug

            image, did_aug = tf.cond(tf.less(tf.random_uniform(shape = ()), P_PXL_SCALE),
                    lambda: random_value_scale(image),
                    lambda: (image, False))
            
            colour_augs['value_scale'] = did_aug
            
            image, did_aug = tf.cond(tf.less(tf.random_uniform(shape = ()), P_S_AND_P),
                    lambda: dropout_salt_and_pepper(image),
                    lambda: (image, False))

            colour_augs['salt_pepper'] = did_aug
            return image, colour_augs

    def random_boxes_translate(self, boxes, ratio, p):
        with tf.name_scope('box_translate'):
            choose = tf.tile(tf.less(tf.random_uniform(shape = (tf.shape(boxes)[0], 1)), p), (1, 4))
            w_h = boxes[:, 2:] - boxes[:, :2]
            rand_pert = tf.random_uniform(tf.shape(w_h), minval = -ratio, maxval = ratio, dtype = tf.float32)
            pert = tf.multiply(rand_pert, w_h)
            boxes_pert = tf.add(boxes, tf.tile(pert, (1,2)))
            return tf.where(choose, boxes_pert, boxes)
    
    def random_boxes_scale(self, boxes, ratio, p):
        with tf.name_scope('box_scale'):
            choose = tf.tile(tf.less(tf.random_uniform(shape = (tf.shape(boxes)[0], 1)), p), (1, 4))
            w_h = boxes[:, 2:] - boxes[:, :2]
            centers = (boxes[:, :2] + boxes[:, 2:])/2
            rand_pert = tf.random_uniform(tf.shape(w_h), minval = 1-ratio, maxval = 1+ratio, dtype = tf.float32)
            w_h_n = tf.multiply(rand_pert, w_h)
            return  tf.where(choose, tf.concat([centers - w_h_n/2.0, centers + w_h_n/2.0], axis = -1), boxes)

    def random_resize(self, img, box, min_scale, max_scale):
        with tf.name_scope('r_resize'):
            r_scale = tf.random_uniform(shape = (), minval = min_scale, maxval = max_scale)
            n_size = tf.cast(tf.cast(tf.shape(img)[:2], tf.float32)*r_scale, tf.int32)
            self.p1 = r_scale
            self.image_sizes = [tf.shape(img), n_size]
            img = tf.image.resize_images(img, n_size)
            box = tf.multiply(box, r_scale)
            return img, box, r_scale, tf.constant(True)

    def build_random_crop(self):
        P_SHOULD_CROP = 0.8
        P_SHOULD_SCALE = 0.7
        SCALE_MIN, SCALE_MAX = 0.75, 1.5
        with tf.name_scope('crop'):
            self.image_in = tf.placeholder(tf.float32, (None, None, 3))
            self.boxes_in = tf.placeholder(tf.float32, (None, 4))
            image, boxes = self.image_in, self.boxes_in

            image, boxes, did_aug = tf.cond(tf.less(tf.random_uniform(shape = ()), P_SHOULD_CROP),
                    lambda: self._random_crop_image(image, boxes),
                    lambda: (image, boxes, False))
            image, boxes, scale, did_scale = tf.cond(
                    tf.less(tf.random_uniform(shape = ()), P_SHOULD_SCALE),
                    lambda: self.random_resize(image, boxes, SCALE_MIN, SCALE_MAX),
                    lambda: (image, boxes, 1.0, False))
            self.post_crop = image, boxes, {'crop': did_aug, 'scale':(did_scale, scale)}
    
    def _random_crop_image(self, image, boxes):
        MIN_OBJ_COVERED = 0.4
        ASPECT_RATIO_RANGE = (0.5, 1.5)
        AREA_RANGE = (0.4, 1.0)
        OVERLAP_THRESH = 0.2
        norm_boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis = 1)
        norm_boxes = norm_boxes/tf.to_float(tf.tile(tf.reshape(tf.shape(image)[:2], (1, 2)), (1, 2)))
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes= tf.expand_dims(norm_boxes, 0),
            min_object_covered=MIN_OBJ_COVERED,
            aspect_ratio_range=ASPECT_RATIO_RANGE,
            area_range=AREA_RANGE,
            max_attempts=100,
            use_image_if_no_bounding_boxes=True
        )
        begin, size, window = sample_distorted_bounding_box
        image_crop = tf.slice(image, begin, size)
        window = tf.squeeze(window, axis=[0, 1])
        # remove boxes that are completely outside cropped image
        boxes_crop, inside_window_ids = self._prune_completely_outside_window(
            norm_boxes, window
        )
        # remove boxes that are two much outside image
        boxes_crop, keep_ids = self._prune_non_overlapping_boxes(
            boxes_crop, tf.expand_dims(window, 0), OVERLAP_THRESH
        )
        # change coordinates of the remaining boxes
        boxes_crop = self._change_coordinate_frame(boxes_crop, window)
        keep_ids = tf.gather(inside_window_ids, keep_ids) # Not strictly needed
        boxes_crop = boxes_crop * tf.to_float(tf.tile(tf.reshape(tf.shape(image_crop)[:2], (1, 2)), (1, 2)))
        boxes_crop = tf.stack([boxes_crop[:, 1], boxes_crop[:, 0], boxes_crop[:, 3], boxes_crop[:,2]], axis = 1)
        return image_crop, boxes_crop, tf.constant(True)

    def intersection(self, boxes1, boxes2):
        """Compute pairwise intersection areas between boxes.
        Arguments:
            boxes1: a float tensor with shape [N, 4].
            boxes2: a float tensor with shape [M, 4].
        Returns:
            a float tensor with shape [N, M] representing pairwise intersections.
        """
        with tf.name_scope('intersection'):

            ymin1, xmin1, ymax1, xmax1 = tf.split(boxes1, num_or_size_splits=4, axis=1)
            ymin2, xmin2, ymax2, xmax2 = tf.split(boxes2, num_or_size_splits=4, axis=1)
            # they all have shapes like [None, 1]

            all_pairs_min_ymax = tf.minimum(ymax1, tf.transpose(ymax2))
            all_pairs_max_ymin = tf.maximum(ymin1, tf.transpose(ymin2))
            intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
            all_pairs_min_xmax = tf.minimum(xmax1, tf.transpose(xmax2))
            all_pairs_max_xmin = tf.maximum(xmin1, tf.transpose(xmin2))
            intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
            # they all have shape [N, M]

            return intersect_heights * intersect_widths

    def area(self, boxes):
        """Computes area of boxes.
        Arguments:
            boxes: a float tensor with shape [N, 4].
        Returns:
            a float tensor with shape [N] representing box areas.
        """
        with tf.name_scope('area'):
            ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
            return (ymax - ymin) * (xmax - xmin)

    def _ioa(self, boxes1, boxes2):
        """Computes pairwise intersection-over-area between box collections.
        intersection-over-area (IOA) between two boxes box1 and box2 is defined as
        their intersection area over box2's area. Note that ioa is not symmetric,
        that is, ioa(box1, box2) != ioa(box2, box1).
        Arguments:
            boxes1: a float tensor with shape [N, 4].
            boxes2: a float tensor with shape [M, 4].
        Returns:
            a float tensor with shape [N, M] representing pairwise ioa scores.
        """
        with tf.name_scope('ioa'):
            intersections = self.intersection(boxes1, boxes2)  # shape [N, M]
            areas = tf.expand_dims(self.area(boxes2), 0)  # shape [1, M]
            return tf.divide(intersections, areas)

    def _prune_completely_outside_window(self, boxes, window):
        """Prunes bounding boxes that fall completely outside of the given window.
        This function does not clip partially overflowing boxes.
        (Taken from Tensorflow Object Detection Core)
        Arguments:
            boxes: a float tensor with shape [M_in, 4].
            window: a float tensor with shape [4] representing [ymin, xmin, ymax, xmax]
                of the window.
        Returns:
            boxes: a float tensor with shape [M_out, 4] where 0 <= M_out <= M_in.
            valid_indices: a long tensor with shape [M_out] indexing the valid bounding boxes
                in the input 'boxes' tensor.
        """
        with tf.name_scope('prune_completely_outside_window'):

            y_min, x_min, y_max, x_max = tf.split(boxes, num_or_size_splits=4, axis=1)
            # they have shape [None, 1]
            win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
            # they have shape []

            coordinate_violations = tf.concat([
                tf.greater_equal(y_min, win_y_max), tf.greater_equal(x_min, win_x_max),
                tf.less_equal(y_max, win_y_min), tf.less_equal(x_max, win_x_min)
            ], axis=1)
            valid_indices = tf.squeeze(
                tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))),
                axis=1
            )
            boxes = tf.gather(boxes, valid_indices)
            return boxes, valid_indices

    def _prune_non_overlapping_boxes(self, boxes1, boxes2, min_overlap=0.0):
        """Prunes the boxes in boxes1 that overlap less than thresh with boxes2.
        For each box in boxes1, we want its IOA to be more than min_overlap with
        at least one of the boxes in boxes2. If it does not, we remove it.
        (Taken from Tensorflow Object Detection Core)
        Arguments:
            boxes1: a float tensor with shape [N, 4].
            boxes2: a float tensor with shape [M, 4].
            min_overlap: minimum required overlap between boxes,
                to count them as overlapping.
        Returns:
            boxes: a float tensor with shape [N', 4].
            keep_inds: a long tensor with shape [N'] indexing kept bounding boxes in the
                first input tensor ('boxes1').
        """
        with tf.name_scope('prune_non_overlapping_boxes'):
            ioa = self._ioa(boxes2, boxes1)  # [M, N] tensor
            ioa = tf.reduce_max(ioa, axis=0)  # [N] tensor
            keep_bool = tf.greater_equal(ioa, tf.constant(min_overlap))
            keep_inds = tf.squeeze(tf.where(keep_bool), axis=1)
            boxes = tf.gather(boxes1, keep_inds)
            return boxes, keep_inds

    def _change_coordinate_frame(self, boxes, window):
        """Change coordinate frame of the boxes to be relative to window's frame.
        (Taken from Tensorflow Object Detection Core)
        Arguments:
            boxes: a float tensor with shape [N, 4].
            window: a float tensor with shape [4].
        Returns:
            a float tensor with shape [N, 4].
        """
        with tf.name_scope('change_coordinate_frame'):

            ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
            ymin -= window[0]
            xmin -= window[1]
            ymax -= window[0]
            xmax -= window[1]

            win_height = window[2] - window[0]
            win_width = window[3] - window[1]
            boxes = tf.stack([
                ymin/win_height, xmin/win_width,
                ymax/win_height, xmax/win_width
            ], axis=1)
            boxes = tf.clip_by_value(boxes, clip_value_min=0.0, clip_value_max=1.0)
            return boxes

    def build_augment(self):
        P_BOX_TRANSLATE = 0.7
        P_BOX_SCALE = 0.7
        P_BOX_TRANS_RATIO = 0.15
        P_BOX_SCALE_RATIO = 0.2
        with tf.name_scope('augment'):
            self.image_in_padded = tf.placeholder(tf.float32, (self.size_out[0], self.size_out[1], 3))
            self.boxes_in_padded = tf.placeholder(tf.float32, (None, 4))
            img, boxes = self.image_in_padded, self.boxes_in_padded

            img, boxes, r_ang = self.random_rotate(img, boxes)
            img, boxes, r_f_lr = self.random_flip_lr(img, boxes)
            img, boxes, r_f_ud = self.random_flip_ud(img, boxes)
            img, r_c_augs = self.random_color_mutation(img)
            boxes = self.random_boxes_translate(boxes, P_BOX_TRANS_RATIO, P_BOX_TRANSLATE) # Jitter by max 0.2 of w, h
            boxes = self.random_boxes_scale(boxes, P_BOX_SCALE_RATIO, P_BOX_SCALE) # Scale by max 0.2 of w, h

            self.params = {'ang': r_ang, 'flip_lr':r_f_lr, 'flip_ud':r_f_ud, 'color_augs':r_c_augs}
            self.image_out = img
            self.boxes_out = boxes
        
    def resize_images(self, imgs, boxes):
        DEBUG = False
        w_n, h_n = self.size_out
        boxes_out = []
        img_out = []
        for i in range(len(imgs)):
            img_cur = imgs[i]
            if DEBUG: print(i)
            if DEBUG: print('Before ', img_cur.shape)
            s_f_y = img_cur.shape[1]/h_n
            s_f_x = img_cur.shape[0]/w_n
            if DEBUG: print(s_f_x, s_f_y)
            if s_f_x > 1 or s_f_y > 1:
                if s_f_y > s_f_x:
                    scale = img_cur.shape[0]/img_cur.shape[1]
                    img_cur = cv2.resize(img_cur, (int(h_n), int(scale*h_n)))
                    boxes[i] = np.array([[int(np.round(val/s_f_y)) for val in z] for z in boxes[i]]).copy()
                else:
                    scale = img_cur.shape[1]/img_cur.shape[0]
                    img_cur = cv2.resize(img_cur, (int(scale*w_n), int(w_n)))
                    boxes[i] = np.array([[int(np.round(val/s_f_x)) for val in z] for z in boxes[i]]).copy()                 
            if DEBUG: print('After ', img_cur.shape)
            y_pad = w_n - img_cur.shape[0]
            y_r = randint(0, y_pad)
            x_pad = h_n - img_cur.shape[1]
            x_r = randint(0, x_pad)
            if DEBUG: print('Padding', y_r, y_pad, x_r, x_pad)
            img_cur = cv2.copyMakeBorder(img_cur, y_r, y_pad - y_r, x_r, x_pad - x_r, cv2.BORDER_CONSTANT, value=(0,0,0))
            if DEBUG: print('Post border ', img_cur.shape)
            img_out.append(img_cur.copy())
            boxes_out.append(np.array([self.correct_bboxes([z[0]+x_r, z[1]+y_r, z[2]+x_r, z[3]+y_r], w_n, h_n) for z in boxes[i]]).copy())
            # self.assert_bboxes(boxes_out[i], boxes[i], [x_r, x_pad, y_r, y_pad])
        return np.array(img_out), boxes_out
 
    def correct_bboxes(self, box, w, h):
        if box[0] < 0: box[0] = 0
        if box[1] < 0: box[1] = 0
        if box[2] > w: box[2] = w
        if box[3] > h: box[3] = h
        if box[0] == box[2]:
            if box[2] == w - 1:
                box[0] -= 1
                box[2] -= 1
            box[2]+=1
        if box[1] == box[3]:
            if box[3] == h - 1:
                box[1] -= 1
                box[3] -= 1
            box[3]+=1
        return box

    def proc_boxes(self, boxes):
        boxes = np.round(boxes)
        box_out = []
        for box in boxes.tolist():
            if box[0] >= box[2] and box[1] >= box[3]:
                box = [box[2], box[3], box[0], box[1]]
            box_out.append(self.correct_bboxes(box, self.size_out[0], self.size_out[1]))
        return np.array(box_out, dtype = np.uint16) # Can be 0-1024

    def augment_batch(self, imgs, lbls):
        imgs_crop = []
        boxes_crop = []
        pre_resize_params_l = []
        imgs_out = []
        boxes_out = []
        aug_out = []
        for i in range(len(imgs)):
            feed_dict = {
                self.image_in: imgs[i],
                self.boxes_in: lbls[i]
            }
            post_crop = self.sess.run(self.post_crop, feed_dict = feed_dict)
            img, boxes, pre_resize_params = post_crop
            imgs_crop.append(img)
            boxes_crop.append(boxes)
            pre_resize_params_l.append(pre_resize_params)
        imgs_crop, boxes_crop  = self.resize_images(imgs_crop, boxes_crop) # Resize to constant size (for input to NN)
        for i in range(len(imgs)):
            feed_dict = {
                self.image_in_padded: np.squeeze(imgs_crop[i]), 
                self.boxes_in_padded: boxes_crop[i]
            }
            img, boxes, params = self.sess.run([self.image_out,
                                               self.boxes_out, 
                                               self.params], feed_dict = feed_dict)
            params['id'] = i
            for key, val in pre_resize_params_l[i].items():
                params[key] = val
            imgs_out.append(img)
            boxes_out.append(self.proc_boxes(boxes)) 
            aug_out.append(params)
        return np.array(imgs_out, dtype = np.uint8), boxes_out, aug_out
