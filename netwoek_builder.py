import tensorflow as tf

try:
    from settings import *
except ImportError:
    pass

#####[batch,1, in_width,in_channels]

def build_tensors():

    with tf.name_scope('Input') as scope:
        inputs = tf.placeholder(dtype='float32', shape=[BATCH_SIZE, 1, VEC_SIZE, CHANNELS], name='Input')
    with tf.name_scope('Target') as scope:
        if network_mode == 'reg':
            target_label = tf.placeholder(dtype='float32', shape=[BATCH_SIZE, 1], name="Targets")
        else:
            target_label = tf.placeholder(dtype='float32', shape=[BATCH_SIZE, INTERVALS_NUM], name="Targets")

    print 'Input:', inputs, '\ntarget:', target_label
    return inputs, target_label





def conv2d_with_relu(inputs, filter_width, stride_size, in_channels, out_channels, name):

    with tf.name_scope(name) as scope:
        weight = tf.Variable(
            tf.random_normal(shape=[1, filter_width, in_channels, out_channels], stddev=STDDEV, mean=MEAN),
            name='Weight')
        # print weight
        biases = tf.Variable(tf.random_normal(shape=[out_channels], stddev=STDDEV, mean=MEAN), name='biases')
        # print biases
        tf.summary.histogram(name + '_biases', biases)
        tf.summary.histogram(name + '_weight', weight)

        conv = tf.nn.conv2d(input=inputs, filter=weight, strides=[1, 1, stride_size, 1],
                            padding='SAME',
                            name='conv') + biases  # ,data_format="NWC") NWC formt - [batch,in_width,in_channels]
        print name, '_conv:' ,conv
        relu = tf.nn.relu(conv, name='relu')
        print name , '_relu:', relu
        return relu


def conv2d(inputs, filter_width, stride_size, in_channels, out_channels, name):

    with tf.name_scope(name) as scope:
        weight = tf.Variable(
            tf.random_normal(shape=[1, filter_width, in_channels, out_channels], stddev=STDDEV, mean=MEAN),
            name='Weight')
        # print weight
        biases = tf.Variable(tf.random_normal(shape=[out_channels], stddev=STDDEV, mean=MEAN), name='biases')
        # print biases
        tf.summary.histogram(name + '_biases', biases)
        tf.summary.histogram(name + '_weight', weight)

        conv = tf.nn.conv2d(input=inputs, filter=weight, strides=[1, 1, stride_size, 1],
                            padding='SAME',
                            name='conv') + biases  # ,data_format="NWC") NWC formt - [batch,in_width,in_channels]
        print name, ':', conv
        return conv


def relu(input, name):

    with tf.name_scope(name) as scope:
        relu = tf.nn.relu(input, name=name)
        print name, ':', relu
        return relu


def maxpooling(input, window_size, stride_size, name):

    with tf.name_scope(name) as scope:
        maxpool = tf.nn.max_pool(input, ksize=[1, 1, window_size, 1],
                                 strides=[1, 1, stride_size, 1], padding="SAME", name='maxpool')
        print name, ':', maxpool
        return maxpool


def dropout(input, DROPOUT_SIZE, name):

    with tf.name_scope(name) as scope:
        dropout = tf.nn.dropout(input, DROPOUT_SIZE, name=name)
        print name, ':', dropout
        return dropout



def fc(input_layer, output_size, name):

    with tf.name_scope(name) as scope:
        input_list = input_layer.get_shape().as_list()
        unit_size = input_list[-1] * input_list[-2] * input_list[-3]
        flat_layer = tf.reshape(input_layer, [-1, unit_size], name=name + '_flat')
        weight = tf.Variable(tf.random_normal([unit_size, output_size], stddev=STDDEV, mean=MEAN), name='weight')
        biases = tf.Variable(tf.random_normal([BATCH_SIZE, output_size], stddev=STDDEV, mean=MEAN), name='biases')
        tf.summary.histogram(name + '_biases', biases)
        tf.summary.histogram(name + '_weidht', weight)
        dense = tf.matmul(flat_layer, weight) + biases
        print name ,'_input:', flat_layer, '\n', name, ':', dense
        return dense


def create_optimization_reg(prediction, target_label):

    with tf.name_scope('Optimization_Block') as scope:
        cost = tf.nn.l2_loss(prediction - target_label)
        cost = tf.reduce_mean(cost)  # need to check
        tf.summary.scalar('cost', cost)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
        #print "cost:", cost.shape, cost
        print "optimizer:", optimizer
        return cost, optimizer


def create_optimization_class(input, target_label):

    with tf.name_scope('Optimization_Block') as scope:
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=input, labels=target_label)
        cost = tf.reduce_mean(cost)
        tf.summary.scalar('cost', cost)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
        #print "cost:", cost.shape, cost
        print "optimizer:", optimizer
        return cost, optimizer


def build_cnn_model(mode):

    global prediction_class_layer  # check
    input, target_label = build_tensors()
    # conv(inputs,filter, stride, channels, output)+relu
    # maxpooling(input, window_size, stride_size, channels):
    # dropout(input, DROPOUT_SIZE):
    conv1 = conv2d_with_relu(input, 3, 1, 1, 8, 'conv1')
    maxpool1 = maxpooling(conv1, 3, 3, 'maxpool1')
    conv2 = conv2d_with_relu(maxpool1, 3, 1, 8, 32, 'conv2')
    conv3 = conv2d_with_relu(conv2, 3, 1, 32, 64, 'conv3')
    conv4 = conv2d_with_relu(conv3, 3, 1, 64, 64, 'conv4')
    conv5 = conv2d_with_relu(conv4, 3, 1, 64, 64, 'conv5')
    conv6 = conv2d_with_relu(conv5, 3, 1, 64, 64, 'conv6')
    #dropout1 = dropout(conv6, 0.5, 'dropout1')
    if network_mode == 'reg':
        prediction = fc(conv6, 1, 'fc1')
        prediction_class = prediction  # only for
    else:
        prediction_class_layer = fc(conv6, INTERVALS_NUM, 'fc1')
        prediction_class = tf.nn.softmax(prediction_class_layer, name='softmax')
        prediction = tf.argmax(prediction_class, axis=1, name='ardmax')

    if mode == "test":
        return input, target_label, prediction, prediction_class
    else:
        if network_mode == 'reg':
            cost, optimizer = create_optimization_reg(prediction, target_label)
        else:
            cost, optimizer = create_optimization_class(prediction_class_layer, target_label)
    print "END OF BUILD CNN MODEL"
    return input, target_label, cost, optimizer, prediction, prediction_class


def build_cnn_model_inception(mode):
    global prediction_class_layer  # check
    input, target_label = build_tensors()
    # conv(inputs,filter, stride, channels, output)+relu
    # maxpooling(input, window_size, stride_size, channels):
    # dropout(input, DROPOUT_SIZE):
    conv1_1 = conv2d(input, 3, 1, 1, 8, 'conv1_1')
    conv1_2 = conv2d(input, 5, 1, 1, 8, 'conv1_2')
    conv1_3 = conv2d(input, 1, 1, 1, 8, 'conv1_3')
    relu1 = relu(tf.concat([conv1_1, conv1_2, conv1_3], -1, 'concat1'),'relu1')
    conv2_1 = conv2d(relu1, 3, 1, 24, 24, 'conv2_1')
    conv2_2 = conv2d(relu1, 5, 1, 24, 24, 'conv2_2')
    conv2_3 = conv2d(relu1, 1, 1, 24, 24, 'conv2_3')
    relu2 = relu(tf.concat([conv2_1, conv2_2, conv2_3], -1, 'concat2'), 'relu2')
    maxpool1 = maxpooling(relu2, 3, 3, 'maxpool1')
    conv3_1 = conv2d(maxpool1, 3, 1, 72, 24, 'conv3_1')
    conv3_2 = conv2d(maxpool1, 5, 1, 72, 24, 'conv3_2')
    conv3_3 = conv2d(maxpool1, 1, 1, 72, 24, 'conv3_3')
    relu3 = relu(tf.concat([conv3_1, conv3_2, conv3_3], -1, 'concat3'), 'relu3')
    conv4_1 = conv2d(relu3, 3, 1, 72, 24, 'conv4_1')
    conv4_2 = conv2d(relu3, 5, 1, 72, 24, 'conv4_2')
    conv4_3 = conv2d(relu3, 1, 1, 72, 24, 'conv4_3')
    relu4 = relu(tf.concat([conv4_1, conv4_2, conv4_3], -1, 'concat4'), 'relu4')
    maxpool2 = maxpooling(relu4, 3, 3, 'maxpool2')
    conv5_1 = conv2d(maxpool2, 3, 1, 72, 24, 'conv5_1')
    conv5_2 = conv2d(maxpool2, 5, 1, 72, 24, 'conv5_2')
    conv5_3 = conv2d(maxpool2, 1, 1, 72, 24, 'conv5_3')
    relu5 = relu(tf.concat([conv5_1, conv5_2, conv5_3], -1, 'concat5'), 'relu5')
    conv6_1 = conv2d(relu5, 3, 1, 72, 24, 'conv6_1')
    conv6_2 = conv2d(relu5, 5, 1, 72, 24, 'conv6_2')
    conv6_3 = conv2d(relu5, 1, 1, 72, 24, 'conv6_3')
    relu6 = relu(tf.concat([conv6_1, conv6_2, conv6_3], -1, 'concat6'), 'relu6')
    dropout1 = dropout(relu6, 0.5, 'dropout1')
    if network_mode == 'reg':
        prediction = fc(dropout1, 1, 'fc1')
        prediction_class = prediction  # only for
    else:
        prediction_class_layer = fc(dropout1, INTERVALS_NUM, 'fc1')
        prediction_class = tf.nn.softmax(prediction_class_layer, name='softmax')
        prediction = tf.argmax(prediction_class, axis=1, name='argmax')

    if mode == "test":
        return input, target_label, prediction, prediction_class
    else:
        if network_mode == 'reg':
            cost, optimizer = create_optimization_reg(prediction, target_label)
        else:
            cost, optimizer = create_optimization_class(prediction_class_layer, target_label)
    print "END OF BUILD CNN MODEL"
    return input, target_label, cost, optimizer, prediction, prediction_class
