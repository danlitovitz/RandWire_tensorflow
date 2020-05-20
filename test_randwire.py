

import tensorflow as tf
import numpy as np
import argparse
from dataset import iter_utils

# main function for test
def main():
    args = argparse.ArgumentParser()
    args.__dict__['class_num'] = 10
    args.__dict__['checkpoint_dir'] = code_directory+'checkpoint_model{}'.format(2)
    args.__dict__['checkpoint_dir'] = './checkpoint/best'
    args.__dict__['test_record_dir'] = './dataset/cifar10/test.tfrecord'
    args.__dict__['batch_size'] = 256

    with tf.Session() as sess:
        #restoring network and weight data
        try:
            saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(args.checkpoint_dir) + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
        except:
            print('failed to load network and checkpoint')
            return
        print('network graph and checkpoint restored')

        # create batch iterator

        test_iterator = iter_utils.batch_iterator(args.test_record_dir, None, args.batch_size, training=False, drop_remainder=False)
        test_images_batch, test_labels_batch = test_iterator.get_next()
        sess.run(test_iterator.initializer)

        graph = tf.get_default_graph()

        # get tensor for feed forward
        images = graph.get_tensor_by_name('images:0')
        labels = graph.get_tensor_by_name('labels:0')
        prediction = graph.get_tensor_by_name('accuracy/prediction:0')
        training = graph.get_tensor_by_name('training:0')

        predictions = 0
        dataset_size = 0

        # test
        while True:
            try:
                test_images, test_labels = sess.run([test_images_batch, test_labels_batch])
                test_labels = np.eye(args.class_num)[test_labels]
                prediction_ = sess.run(prediction, feed_dict={images: test_images, labels: test_labels, training: False})
                predictions += np.sum(prediction_.astype(int))
                dataset_size += len(prediction_)
                print('\r{0} done'.format(dataset_size), end='')
            except tf.errors.OutOfRangeError:
                print('\n')
                break

        print('test accuracy: ', (predictions / dataset_size) * 100, '%')

if __name__ == '__main__':
    main()
