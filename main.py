#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf_graph = tf.get_default_graph()

    #Fetch all the above senson name from tensor graph
    tf_graph.get_tensor_by_name(vgg_input_tensor_name)
    tf_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    tf_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    tf_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    tf_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor_name, vgg_keep_prob_tensor_name, vgg_layer3_out_tensor_name, vgg_layer4_out_tensor_name,\
           vgg_layer7_out_tensor_name
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
    '''
        The first step is to have 1x1 connected layer from encoded vgg layer 7 output
        using convolve 2d function
    '''
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, filter=num_classes, kernel_size=1, padding='same',
                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    #To decode unsample layer 8 to match the vgg_layer_4 in order to add skip connection
    unsampled_layer7 = tf.layers.conv2d_transpose(layer7_1x1, filter=num_classes, kernel_size=4, strides=(2, 2),
                                                 padding='same',kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    #Let's convolve vgg layer 4 to 1 x 1 so that we can add it to layer 8
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, filter=num_classes, kernel_size=1, padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    #add skip connection
    skip_connection_1 = tf.add(layer4_1x1,unsampled_layer7)

    #Let's unsample again so we can add it to vgg_layer3_out
    unsampled_skip1 = tf.layers.conv2d_transpose(skip_connection_1, filter=num_classes, kernel_size=4, strides=(2, 2),
                                                 padding='same', kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # Let's convolve vgg layer 3 output to 1 x 1 so that we can add
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, filter=num_classes, kernel_size=1, padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    #add skip connection
    skip_connection_2 = tf.add(unsampled_skip1, layer3_1x1)

    #final unsampling
    unsampled_skip2 = tf.layers.conv2d_transpose(skip_connection_2, filter=num_classes, kernel_size=16, strides=(8, 8),
                                                 padding='same', kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))




    return unsampled_skip2
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
    ''' In this method lets define the loss and optimizer so that we can train the neural network
        the goal here is to assign each pixel an appropriate class using cross entropy loss
    '''
    #We need to reshape the output sensor from 4d to 2D
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    #Let's reshape the label as well so that we can compute cross entropy loss
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    '''add loss manually here based on the suggestion 
     https://stackoverflow.com/questions/46615623/do-we-need-to-add-the-regularization-loss
     -into-the-total-loss-in-tensorflow-mode
     '''

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 1e-3  # Choose an appropriate one.
    loss = cross_entropy_loss + reg_constant * sum(reg_losses)

    #Use Adam optmizier and minimize the loss
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
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

    sess.run(tf.global_variables_initializer())

    print("Training the network \n")

    for i in range(epochs):
        print("Current Iteration count : {}".format(i))
        total_loss = 0
        for image, label in get_batches_fn(batch_size):
            loss, _  = sess.run([cross_entropy_loss, train_op],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5,
                                          learning_rate: 0.001})
            total_loss += loss

        print("Loss: = {:.3f}\n".format(loss))
    
    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        epochs = 50
        batch_size = 5

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
