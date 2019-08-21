import os
import logging
import tensorflow as tf

from lib.setup import params_setup, logging_config_setup, config_setup
from lib.model_utils import create_graph, load_weights, print_num_of_trainable_parameters, save_weights
from lib.train import train
from lib.test import test
from lib.predict import predict

#tf.enable_eager_execution()

def main():
    para = params_setup()
    logging_config_setup(para)

    logging.info('Creating graph')
    graph, model, data_generator = create_graph(para)

    with tf.Session(config=config_setup(), graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        logging.info('Loading weights')
        load_weights(para, sess, model)
        print_num_of_trainable_parameters()

        try:
            if para.mode == 'train':
                logging.info('Started training')
                train(para, sess, model, data_generator)
                save_weights(sess, model, '', 'testModel3')
            elif para.mode == 'validation':
                logging.info('Started validation')
                test(para, sess, model, data_generator)
            elif para.mode == 'test':
                logging.info('Started testing')
                test(para, sess, model, data_generator)
            elif para.mode == 'predict':
                logging.info('Predicting')
                predict(para, sess, model, './data/solar-energy3/solar_predict.txt', 10)


        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            print('Stop')


if __name__ == '__main__':
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    main()
