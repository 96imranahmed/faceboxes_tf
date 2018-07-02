import tensorflow as tf
import numpy as np
import cv2
from model import FaceBox
import anchors
import pickle
import data
import multiprocessing

def count_number_trainable_params(scope = ""):
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    vars_chk = None
    if scope == "": 
        vars_chk = tf.trainable_variables()
    else: 
        vars_chk = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    for trainable_variable in vars_chk:
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shape.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    np.set_printoptions(suppress=True)
    data_train_source = './wider_train.p'
    data_test_source = './wider_test.p'
    data_train_dir = '../WIDER/train_images/'
    data_test_dir = '../WIDER/test_images/'
    save_f = './models/'
    model_name = 'facebox'
    PRINT_FREQ = 500
    TEST_FREQ = 1000
    SAVE_FREQ = 5000
    BATCH_SIZE = 15
    IM_S = 1024
    IM_CHANNELS = 3
    N_WORKERS = 12
    MAX_PREBUFF_LIM = 20
    IOU_THRESH = 0.5
    USE_NORM = True
    CONFIG = [[1024, 1024, 32, 32, 32, 32, 4], 
            [1024, 1024, 32, 32, 64, 64, 2],
            [1024, 1024, 32, 32, 128, 128, 1],
            [1024, 1024, 64, 64, 256, 256, 1],
            [1024, 1024, 128, 128, 512, 512, 1]]
    IS_AUG = False
    # NOTE: SSD variances are set in the anchors.py file
    boxes_vec, boxes_lst, stubs = anchors.get_boxes(CONFIG, normalised = USE_NORM)
    tf.reset_default_graph()

    train_data = pickle.load(file = open(data_train_source, 'rb'))
    test_data = pickle.load(file = open(data_test_source, 'rb'))
    
    svc_train = None
    if IS_AUG:
        augmenter_dict = {'lim': MAX_PREBUFF_LIM, 'n':N_WORKERS, 'b_s':BATCH_SIZE}
        svc_train = data.DataService(train_data, True, data_train_dir, (1024, 1024), augmenter_dict, normalised = USE_NORM)
        print('Starting augmenter...')
        svc_train.start()
        print('Running model...')
    else:
        svc_train = data.DataService(train_data, False, data_train_dir, (1024, 1024), normalised = USE_NORM)
    svc_test = data.DataService(test_data, False, data_test_dir, (1024, 1024), normalised = USE_NORM)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print('Building model...')
        fb_model = FaceBox(sess, (BATCH_SIZE, IM_S, IM_S, IM_CHANNELS), boxes_vec, normalised = USE_NORM)
        print('Num params: ', count_number_trainable_params())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=2)
        try:
            ckpt = tf.train.get_checkpoint_state(save_f + model_name)
            if ckpt is None:
                raise IOError('No valid save file found')
            print('#####################')
            print(ckpt.model_checkpoint_path)
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Succesfully loaded saved model')
        except IOError:
            print('Model not found - using default initialisation!')
            sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./logs', sess.graph)
        i = 0
        train_mAP_pred = []
        train_loss = []
        test_mAP_pred = []
        while (1<2):
            print(' Iteration ', i, '                                                ', end = '\r')
            i+=1
            imgs, lbls = None, None
            if IS_AUG: 
                imgs, lbls = svc_train.pop()
            else:
                imgs, lbls = svc_train.random_sample(BATCH_SIZE)
            pred_confs, pred_locs, loss, summary, mAP = fb_model.train_iter(boxes_vec, imgs, lbls)
            train_loss.append(loss)
            train_mAP_pred.append(mAP)
            writer.add_summary(summary, i)
            if i%PRINT_FREQ == 0:
                print("")
                print('Iteration: ', i)
                print('Mean train loss: ', np.mean(train_loss))
                print('Mean train mAP: ', np.mean(train_mAP_pred))
                train_mAP_pred = []
                train_loss = []
            if i%TEST_FREQ == 0:
                for j in range(100):
                    imgs, lbls = svc_test.random_sample(BATCH_SIZE)
                    pred_confs, pred_locs = fb_model.test_iter(imgs)
                    pred_boxes = anchors.decode_batch(boxes_vec, pred_locs, pred_confs)
                    test_mAP_pred.append(anchors.compute_mAP(imgs, lbls, pred_boxes, normalised = USE_NORM))
                print('Mean test mAP: ', np.mean(test_mAP_pred))
                test_mAP_pred = []
            if i%SAVE_FREQ == 0:
                print('Saving model...')
                saver.save(sess, save_f + model_name, global_step = i)

    

