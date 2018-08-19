import sys
import cv2
import tensorflow as tf
from model import FaceBox
import anchors
import pickle
import data
import numpy as np
import os

def lighting_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def main(argv):
    # Constants and misc.
    save_f = './models/'
    model_name = 'facebox'
    WIDTH_DES = 1024
    HEIGHT_DES = 1024
    IM_CHANNELS = 3
    BATCH_SIZE = 1
    USE_NORM = True
    ANCHOR_CONFIG = [[1024, 1024, 32, 32, 32, 32, 4], 
            [1024, 1024, 32, 32, 64, 64, 2],
            [1024, 1024, 32, 32, 128, 128, 1],
            [1024, 1024, 64, 64, 256, 256, 1],
            [1024, 1024, 128, 128, 512, 512, 1]]
    boxes_vec, boxes_lst, stubs = anchors.get_boxes(ANCHOR_CONFIG, normalised = USE_NORM)

    # Setup tensorflow and model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Force on CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force on CPU
    config = tf.ConfigProto()
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        model = FaceBox(sess, (BATCH_SIZE, WIDTH_DES, HEIGHT_DES, IM_CHANNELS), boxes_vec, normalised = USE_NORM)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=2)
        try:
            ckpt = tf.train.get_checkpoint_state(save_f)
            if ckpt is None:
                raise IOError('No valid save file found')
            print('#####################')
            print(ckpt.model_checkpoint_path)
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Succesfully loaded saved model')
        except IOError:
            print('Model not found - using random init')
            sess.run(tf.global_variables_initializer())
        
        # Setup webcam feed
        vid_in = cv2.VideoCapture(0)
        ret = True
        
        #Loop through video data
        while (ret == True):
            ret, frame = vid_in.read()
            r = WIDTH_DES / frame.shape[1]
            dim_des = (int(WIDTH_DES), int(frame.shape[0] * r))
            frame = cv2.resize(frame, dim_des, interpolation = cv2.INTER_LANCZOS4)
            frame_padded = lighting_balance(frame)
            frame_padded = cv2.copyMakeBorder(frame, 0, HEIGHT_DES - frame.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            pred_confs, pred_locs = model.test_iter(np.expand_dims(frame_padded, axis = 0))
            pred_boxes = anchors.decode_batch(boxes_vec, pred_locs, pred_confs, min_conf=0.25)[0]
            pred_boxes[pred_boxes < 0] = 0
            pred_boxes[:, [0, 2]][pred_boxes[:, [0, 2]] > WIDTH_DES] = WIDTH_DES
            pred_boxes[:, [1, 3]][pred_boxes[:, [1, 3]] > HEIGHT_DES] = HEIGHT_DES 
            h, w = HEIGHT_DES, WIDTH_DES
            for box in pred_boxes.tolist():
                if USE_NORM:
                    cv2.rectangle(frame, (int(box[0]*w),int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (0,255,0), 3)
                else:
                    cv2.rectangle(frame, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 3)    
            cv2.imshow('Webcam', frame)
            cv2.waitKey(1)
        vid_in.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
