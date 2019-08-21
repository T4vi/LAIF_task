import logging
import numpy as np 

def predict(para, sess, model, file_path, samples):
    sess.run(data_generator.iterator.initializer)

	#batch_size = 1, mode = 'predict', samples > 0 < ?, initial_weights

    logging.info(f'Loading data from {file_path}')

    #dataset = tf.data.TFRecordDataset(file_path)
    #dataset = dataset.map(self._decode)
    
    raw_dat = np.loadtxt(file_path, delimiter=',')

    para.input_size = para.output_size = raw_dat.shape[1]

    raw_dat = raw_dat[-para.attention_len:]
    #raw_dat = tf.transpose(raw_dat)
    raw_dat = tf.expand_dims(raw_dat, 0)

    for k in range(samples):
    	sess.run(self.iterator.initializer)
    	outputs = sess.run(fetches=[model.all_rnn_outputs], feed_dict={ x: raw_dat })
    	raw_dat = tf.concat([raw_dat[:-1], outputs], 0)

    return tf.concat([raw_dat[-samples+1:], outputs])