#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import sys

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()

    return graph.get_tensor_by_name('image_input:0'),\
        graph.get_tensor_by_name('keep_prob:0'),\
        graph.get_tensor_by_name('layer3_out:0'),\
        graph.get_tensor_by_name('layer4_out:0'),\
        graph.get_tensor_by_name('layer7_out:0')


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    k_init = tf.truncated_normal_initializer(stddev=0.01)

    # output of layer 7
    conv_1x1 = tf.layers.conv2d(
        vgg_layer7_out, num_classes, 1, padding='same', kernel_regularizer=reg, kernel_initializer=k_init)
    output = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(
        2, 2), padding='same', kernel_regularizer=reg, kernel_initializer=k_init)

    # add in skip layers
    pool4_out_scaled = tf.multiply(
        vgg_layer4_out, 0.01, name='pool4_out_scaled')
    conv_1x1 = tf.layers.conv2d(
        pool4_out_scaled, num_classes, 1, padding='same', kernel_regularizer=reg, kernel_initializer=k_init)
    conv_1x1 = tf.add(output, conv_1x1)
    output = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(
        2, 2), padding='same', kernel_regularizer=reg, kernel_initializer=k_init)

    pool3_out_scaled = tf.multiply(
        vgg_layer3_out, 0.0001, name='pool3_out_scaled')
    conv_1x1 = tf.layers.conv2d(
        pool3_out_scaled, num_classes, 1, padding='same', kernel_regularizer=reg, kernel_initializer=k_init)
    conv_1x1 = tf.add(output, conv_1x1)
    output = tf.layers.conv2d_transpose(conv_1x1, num_classes, 16, strides=(
        8, 8), padding='same', kernel_regularizer=reg, kernel_initializer=k_init)

    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    training_operation = optimizer.minimize(cross_entropy_loss)

    return logits, training_operation, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    epoch = 0
    total_loss = sys.float_info.max
    last_loss = total_loss
    last_last_loss = last_loss
    while last_last_loss >= total_loss:
        last_last_loss = last_loss
        last_loss = total_loss
        epoch += 1
        print("Epoch {} ...".format(epoch))
        total_loss = 0
        count = 0
        for image, label in get_batches_fn(batch_size):
            feed_dict = {input_image: image,
                         correct_label: label,
                         keep_prob: .60,
                         learning_rate: .00005}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict)
            total_loss += loss
            count += 1
            print("Loss = {:.3f}".format(loss))
            print()
        print("total_loss = {:.3f}".format(total_loss/count))
        print()


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    batch_size = 25

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(
            sess, vgg_path)

        final_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        correct_label = tf.placeholder(
            tf.int32, shape=[None, None, None, num_classes], name='correct_label')
        logits, training_operation, cross_entropy_loss = optimize(
            final_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, batch_size, get_batches_fn, training_operation,
                 cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(
            runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
