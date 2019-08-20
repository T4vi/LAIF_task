from copy import deepcopy
import os
import logging
import tensorflow as tf

from lib.model import PolyRNN


def create_data_generator(para):
    if para.data_set == 'lpd5':
        from lib.data_generator import LPD5DataGenerator
        return LPD5DataGenerator(para)
    elif para.data_set == 'muse':
        from lib.data_generator import MuseDataGenerator
        return MuseDataGenerator(para)
    elif para.mts:
        from lib.data_generator import TimeSeriesDataGenerator
        return TimeSeriesDataGenerator(para)
    else:
        raise ValueError('data_set {} is unknown'.format(para.data_set))


def create_graph(para):
    graph = tf.Graph()
    with graph.as_default():
        initializer = tf.random_uniform_initializer(-para.init_weight,
                                                    para.init_weight)
        data_generator = create_data_generator(para)
        with tf.variable_scope('model', initializer=initializer):
            model = PolyRNN(para, data_generator)
    return graph, model, data_generator


def create_valid_graph(para):
    valid_para = deepcopy(para)
    valid_para.mode = 'validation'

    valid_graph, valid_model, valid_data_generator = create_graph(valid_para)
    return valid_para, valid_graph, valid_model, valid_data_generator

#loads saved weights into model. 
#If initial_weights is set, the first iteration will start with those weights. Continues with checkpoints
def load_weights(para, sess, model):
    if para.first_epoch and para.initial_weights != '':
        para.first_epoch = False
        logging.info('Loading initial model from %s', para.initial_weights)
        model.saver.restore(sess, para.initial_weights)
    else:
        ckpt = tf.train.get_checkpoint_state(para.model_dir)
        if ckpt:
            logging.info('Loading saved model from %s', ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logging.info('Loading model with fresh parameters')
            sess.run(tf.global_variables_initializer())


def save_model(para, sess, model):
    [global_step] = sess.run([model.global_step])
    logging.debug(f'Saving model, step {global_step}')
    checkpoint_path = os.path.join(para.model_dir, "model.ckpt")
    model.saver.save(sess, checkpoint_path, global_step=global_step)

#save model in folder pointed at by path as <name>.ckpt
def save_weights(sess, model, path, name):
    #should add a timestamp or some unique ID
    path = os.path.join(path + name + ".w")
    logging.debug(f'Saving model {name} at {path}')
    model.saver.save(sess, path)

def print_num_of_trainable_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    logging.info('# of trainable parameters: %d' % total_parameters)

#cleans learnt model
def cleanup_train(sess, model):
    logging.info('Clearing model')
    sess.run(tf.global_variables_initializer())
