import tensorflow as tf
import numpy as np
from model import FaceBox
import anchors
import pickle

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

data_train_source = './wider_train.p'
save_f = './models/autoencoder'
PRINT_FREQ = 200
TEST_FREQ = 200
SAVE_FREQ = 2000
BATCH_SIZE = 5
IM_S = 1024
IM_CHANNELS = 3
IOU_THRESH = 0.5
CONFIG = [[1024, 1024, 32, 32, 32, 32, 4], 
          [1024, 1024, 32, 32, 64, 64, 2],
          [1024, 1024, 32, 32, 128, 128, 1],
          [1024, 1024, 64, 64, 256, 256, 1],
          [1024, 1024, 128, 128, 512, 512, 1]] 
# NOTE: SSD variances are set in the anchors.py file
boxes_vec, boxes_lst, stubs = anchors.get_boxes(CONFIG)
tf.reset_default_graph()

train_data = pickle.load(file = open(data_train_source, 'rb'))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    print('Building model...')
    fb_model = FaceBox(sess, (BATCH_SIZE, IM_S, IM_S, IM_CHANNELS), boxes_vec.shape, IOU_THRESH)
    print('Num params: ', count_number_trainable_params())
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
        print('Model not found - using default initialisation!')
        sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logs', sess.graph)