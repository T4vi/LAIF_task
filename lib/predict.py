import logging
import os
import tensorflow as tf
import numpy as np 

## @args:
#  data_generator: prediction data generator (data to predict on/from)
#  file_path: prediciton data file path
#  samples: # of samples to predict
## @return [samples x outputsize] vector of predictions
def predict(para, sess, model, data_generator, file_path, samples):
    sess.run(data_generator.iterator.initializer)

    #batch_size = 1, mode = 'predict', samples > 0 < ?, initial_weights

    logging.info(f'Loading data from {file_path}')

    #dataset = tf.data.TFRecordDataset(file_path)
    #dataset = dataset.map(self._decode)
    
    # raw_dat = np.loadtxt(file_path, delimiter=',')

    # para.input_size = para.output_size = raw_dat.shape[1]

    # raw_dat = raw_dat[-para.attention_len:]
    # #raw_dat = tf.transpose(raw_dat)
    # raw_dat = tf.expand_dims(raw_dat, 0)

    predictions = np.zeros(samples * para.input_size)
    predictions = np.reshape(predictions, (samples, para.input_size))
    for k in range(samples):
        print(k)
        sess.run(data_generator.iterator.initializer)
        inputs, outputs = sess.run(fetches=[model.rnn_inputs, model.all_rnn_outputs])
        
        # append to file to include in next iteration
        _append_results(para, inputs, outputs)

        #append to output vector
        predictions[k] = outputs

    print(np.asarray(predictions).shape)

    # return final predictions
    return outputs


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# this could be done in data_generator to clean this file
def _append_results(para, inputs, outputs):
    print('Appending results')
    filename = "../../test.tfrecords"   ##make this a param
    if os.path.exists(filename):
        print('path exists')

    # dataset = tf.data.TFRecordDataset(filename)
    # dataset = dataset.map(_decode_predict)

    # print(para.attention_len, para.input_size)

    # print('Reading data')
    #Read existing data
    # filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    # example = tf.parse_single_example(
    #     serialized_example,
    #     features={
    #         "x":
    #         tf.FixedLenFeature([para.attention_len, para.input_size],
    #                            tf.float32),
    #         "y":
    #         tf.FixedLenFeature([para.attention_len], tf.float32),
    #     })

    # x = tf.to_float(tf.reshape(example['x'], (para.attention_len, para.input_size)))
    # print(x)

    # print(np.asarray(x).shape)
    # print(np.asarray(outputs).shape)
    # inputs = np.asarray(inputs)

    # outputs = np.asarray(outputs, dtype=np.float32)
    # outputs.flatten()
    # outputs = list(outputs)
    # print(inputs)
    # print(outputs)

    inputs = np.reshape(inputs, (16, 137))
    inputs = inputs[-para.attention_len+1:]
    print(np.asarray(inputs).shape)
    inputs = np.append(inputs, outputs)
    inputs = np.reshape(inputs, (16, 137))

    inputs = list(inputs)
    # inputs = [[ii for ii in range(para.input_size)] for _ in range(para.attention_len)]
    #Really doesnt matter
    outputs = [i for i in range(para.input_size)]

    print(np.asarray(inputs).shape)
    print(np.asarray(outputs).shape)

    # print()

    print('Writing data')
    #append predicted output
    # x = x[-para.attention_len+1:]
    # x = [*x, *output]
    # x = np.asarray(x).flatten()
    # y = tf.tile(tf.expand_dims(tf.to_float(example["y"]), 0), [para.attention_len, 1])
    # print(x)
    # print(x.shape)
    #Write new data
    with tf.python_io.TFRecordWriter(filename) as record_writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "x":_float_list_feature(np.asarray(inputs).flatten()),
                    "y":_float_list_feature(outputs),
                }))
        record_writer.write(example.SerializeToString())