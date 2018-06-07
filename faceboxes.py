import tensorflow as tf
import numpy as np
from model import FaceBox

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

save_f = './models/autoencoder'
PRINT_FREQ = 200
TEST_FREQ = 200
SAVE_FREQ = 2000
tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    print('Building model...')
    fb_model = FaceBox(sess, (5, 1024, 1024, 3), None)
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