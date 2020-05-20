import tensorflow as tf
import argparse
import numpy as np
from network import RandWire
from dataset import iter_utils
import os

# main function for training
def main():
    # TODO
    tf.compat.v1.reset_default_graph()
    FOLDERNAME = 'cs231n/project/RandWire_tensorflow'
    code_directory = '/content/drive/My Drive/{}/'.format(FOLDERNAME)

    dataset_dir = 'dataset/cifar10/train.tfrecord/'

    args = argparse.ArgumentParser()
    args.__dict__['class_num'] = 10
    args.__dict__['image_shape'] = [32, 32, 3]
    args.__dict__['stages'] = 4
    args.__dict__['channel_count'] = 78
    args.__dict__['graph_model'] = 'ws'
    args.__dict__['graph_param'] = [32.0, 4.0, 0.75]
    args.__dict__['dropout_rate'] = 0.2
    args.__dict__['learning_rate'] = 1e-1
    args.__dict__['momentum'] = 0.9
    args.__dict__['weight_decay'] = 1e-4
    args.__dict__['train_set_size'] = 50000
    args.__dict__['val_set_size'] = 10000
    args.__dict__['batch_size'] = 100
    args.__dict__['epochs'] = 100
    args.__dict__['checkpoint_dir'] = code_directory+'checkpoint_model{}'.format(2)
    args.__dict__['checkpoint_name'] = 'randwire_cifar10'
    args.__dict__['train_record_dir'] = code_directory+'dataset/cifar10/train.tfrecord'
    args.__dict__['val_record_dir'] = code_directory+'dataset/cifar10/test.tfrecord'

    images = tf.compat.v1.placeholder('float32', shape=[None, *args.image_shape], name='images')  # placeholder for images
    labels = tf.compat.v1.placeholder('float32', shape=[None, args.class_num], name='labels')  # placeholder for labels
    training = tf.compat.v1.placeholder('bool', name='training')  # placeholder for training boolean (is training)
    global_step = tf.compat.v1.get_variable(name='global_step', shape=[], dtype='int64', trainable=False)  # variable for global step
    best_accuracy = tf.compat.v1.get_variable(name='best_accuracy', dtype='float32', trainable=False, initializer=0.0)
    
    steps_per_epoch = round(args.train_set_size / args.batch_size)
    learning_rate = tf.compat.v1.train.piecewise_constant(global_step, [round(steps_per_epoch * 0.5 * args.epochs),
                                                              round(steps_per_epoch * 0.75 * args.epochs)],
                                                [args.learning_rate, 0.1 * args.learning_rate,
                                                 0.01 * args.learning_rate])
    # output logit from NN
    # output = RandWire.my_small_regime(images, args.stages, args.channel_count, args.class_num, args.dropout_rate,
    #                             args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', False, training)
    # output = RandWire.my_regime(images, args.stages, args.channel_count, args.class_num, args.dropout_rate,
    #                             args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', False, training)
    output = RandWire.small_regime(images, args.stages, args.channel_count, args.class_num, args.dropout_rate,
                                args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', False,
                                training)
    # output = RandWire.regular_regime(images, args.stages, args.channel_count, args.class_num, args.dropout_rate,
    #                             args.graph_model, args.graph_param, args.checkpoint_dir + '/' + 'graphs', training)

    #loss and optimizer
    with tf.compat.v1.variable_scope('losses'):
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output)
        loss = tf.compat.v1.losses.softmax_cross_entropy(labels, output, label_smoothing=0.1)
        loss = tf.reduce_mean(loss, name='loss')
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.compat.v1.trainable_variables()], name='l2_loss')

    with tf.compat.v1.variable_scope('optimizers'):
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=args.momentum, use_nesterov=True)
        #optimizer = tf.train.AdamOptimizer(learning_rate)

        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(loss + l2_loss * args.weight_decay, global_step=global_step)
        train_op = tf.group([train_op, update_ops], name='train_op')

    #accuracy
    with tf.compat.v1.variable_scope('accuracy'):
        output = tf.nn.softmax(output, name='output')
        prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1), name='prediction')
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')

    # summary
    train_loss_summary = tf.compat.v1.summary.scalar("train_loss", loss)
    val_loss_summary = tf.compat.v1.summary.scalar("val_loss", loss)
    train_accuracy_summary = tf.compat.v1.summary.scalar("train_acc", accuracy)
    val_accuracy_summary = tf.compat.v1.summary.scalar("val_acc", accuracy)

    saver = tf.train.Saver()
    best_saver = tf.train.Saver(max_to_keep=1)

    with tf.compat.v1.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.checkpoint_dir + '/log', sess.graph)

        sess.run(tf.global_variables_initializer())
        augmentations = [lambda image, label: iter_utils.pad_and_crop(image, label, args.image_shape, 4), iter_utils.flip]
        train_iterator = iter_utils.batch_iterator(args.train_record_dir, args.epochs, args.batch_size, augmentations, True)
        train_images_batch, train_labels_batch = train_iterator.get_next()
        val_iterator = iter_utils.batch_iterator(args.val_record_dir, args.epochs, args.batch_size)
        val_images_batch, val_labels_batch = val_iterator.get_next()
        sess.run(train_iterator.initializer)
        if args.val_set_size != 0:
            sess.run(val_iterator.initializer)

        # restoring checkpoint
        try:
            checkpoint_filepath = tf.train.latest_checkpoint(args.checkpoint_dir+'/best')
            saver.restore(sess, checkpoint_filepath)
            print('checkpoint restored. train from checkpoint')
        except:
            print('failed to load checkpoint. train from the beginning')

        #get initial step
        gstep = sess.run(global_step)
        init_epoch = round(gstep / steps_per_epoch)
        init_epoch = int(init_epoch)

        for epoch_ in range(init_epoch + 1, args.epochs + 1):

            # train
            while gstep * args.batch_size < epoch_ * args.train_set_size:
                try:
                    train_images, train_labels = sess.run([train_images_batch, train_labels_batch])
                    train_labels = np.eye(args.class_num)[train_labels]
                    gstep, _, loss_, accuracy_, train_loss_sum, train_acc_sum = sess.run(
                        [global_step, train_op, loss, accuracy, train_loss_summary, train_accuracy_summary],
                        feed_dict={images: train_images, labels: train_labels, training: True})
                    print('[global step: ' + str(gstep) + ' / epoch ' + str(epoch_) + '] -> train accuracy: ',
                          accuracy_, ' loss: ', loss_)
                    writer.add_summary(train_loss_sum, gstep)
                    writer.add_summary(train_acc_sum, gstep)
                except tf.errors.OutOfRangeError:
                    break

            predictions = []

            # validation
            if args.val_set_size != 0:
                while True:
                    try:
                        val_images, val_labels = sess.run([val_images_batch, val_labels_batch])
                        val_labels = np.eye(args.class_num)[val_labels]
                        loss_, accuracy_, prediction_, val_loss_sum, val_acc_sum = sess.run(
                            [loss, accuracy, prediction, val_loss_summary, val_accuracy_summary],
                            feed_dict={images: val_images, labels: val_labels, training: False})
                        predictions.append(prediction_)
                        print('[epoch ' + str(epoch_) + '] -> val accuracy: ', accuracy_, ' loss: ', loss_)
                        writer.add_summary(val_loss_sum, gstep)
                        writer.add_summary(val_acc_sum, gstep)
                    except tf.errors.OutOfRangeError:
                        sess.run(val_iterator.initializer)
                        break

            predictions = np.concatenate(predictions)
            print('best: ', best_accuracy.eval(), '\ncurrent: ', np.mean(predictions))
            if best_accuracy.eval() < np.mean(predictions):
                print('save checkpoint')
                best_accuracy = tf.assign(best_accuracy, np.mean(predictions))
                best_saver.save(sess, args.checkpoint_dir + '/best/' + args.checkpoint_name, global_step=global_step)
