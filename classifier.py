from random import shuffle
import pandas as pd
import numpy as np
from data_generator import prepare_data_vectors
from data_generator import prepare_data_vectors_rand
from netwoek_builder import build_cnn_model
from netwoek_builder import build_cnn_model_inception
from data_generator import get_data
import tensorflow as tf
from collections import Counter
import sys
import os
import time
import datetime

try:
    from settings import *
except ImportError:
    pass


def parse_flags():
    mode = raw_input('choose train/test: ')
    while not (mode == 'test' or mode == 'train'):
        mode = raw_input('choose train/test: ')
    return mode


def parse_flags_2():
    try:
        mode = sys.argv[1]
        return mode,
    except IndexError:
        print "\nUSAGE: python classifier.py <train/test>\n"
        quit()


def create_dataset(dataset_path, mode):
    print dataset_path
    if mode == 'test':
        start_data = 1500 - VEC_SIZE
    else:
        start_data = VEC_SIZE
    data = pd.read_csv(dataset_path)
    vec = np.zeros(shape=(len(data["Open"]) - VEC_SIZE + 1, 1, VEC_SIZE, CHANNELS))  # MAYBE ADD CH
    if network_mode == 'reg':
        label = np.zeros(shape=(len(data["Open"]) - VEC_SIZE + 1, CHANNELS))
    else:
        label = np.zeros(shape=(len(data["Open"]) - VEC_SIZE + 1, INTERVALS_NUM))#maybe problem without channels
    if DATA_MODE == 'random':
        list_num, input_vectors, input_label = prepare_data_vectors_rand(data, start_data, vec, label, VEC_SIZE, LIMIT,network_mode)
        data_set = list(get_data(BATCH_SIZE, list_num, input_vectors, input_label))
    else:
        data_set = list(prepare_data_vectors(data, start_data, vec, label, VEC_SIZE, BATCH_SIZE, LIMIT, network_mode))
    if DATA_MODE == 'random_batch':
        shuffle(data_set)
    print "data set size:", len(data_set)
    # print "data_vec: " ,data_set[0][0]
    # print "data_label: ", data_set[0][1]
    return data_set








def run_session(data_set, cost, optimizer, prediction, prediction_class, inputs, target_label, mode, epochs):
    # global err
    err = 0
    summaryMerged = tf.summary.merge_all()
    filename = './summary_log/run' + datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%s')
    writer = tf.summary.FileWriter(filename, sess.graph)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    model_save_path = 'Saved_Model_' + network_shape + '_' + DATA_MODE
    model_name = 'Classifier'
    print "point1"
    if os.path.exists(os.path.join(model_save_path, 'checkpoint')):
        saver.restore(sess, tf.train.latest_checkpoint(model_save_path))

    # print_model(sess)
    #step = 0
    counter_runs = 0
    right = 0
    #counter = Counter()#need to understand
    if mode == "train":
        epoch_err = []
        for epoch in range(epochs):

            for batch in data_set:
                counter_runs += 1
                data, label = batch[0], batch[1]

                pred, pred_class, err, _, sumOut = sess.run([prediction, prediction_class, cost, optimizer, summaryMerged],
                                                feed_dict={inputs: data, target_label: label})
                writer.add_summary(sumOut, counter_runs)
                print 'counter runs: ', counter_runs, '\nerror rate:', str(err)
                print "epoch:", epoch
                #print 'prediction_class', pred_class, '\nprediction', pred
                if network_mode == 'class':
                    right += check_pred(label, pred)
                    show_stats(right, counter_runs)
                #step += 1
                if counter_runs % SAVING_INTERVAL == 0:
                    #print "epoch:", epoch
                    print "saving model..."
                    saver.save(sess, os.path.join(model_save_path, model_name))
                    print "model saved"
                    print "point2"
                    # show_stats(counter)
            epoch_err.append(str(err))
        for i, ep_err in enumerate(epoch_err):
            print "Epoch number: ", i, "Error rate:", ep_err
    elif mode == "test":
        #right = 0
        for batch in data_set:
            counter_runs += 1
            data, label = batch[0], batch[1]
            pred,pred_class = sess.run([prediction, prediction_class], feed_dict={inputs: data, target_label: label})
            print 'counter runs: ', counter_runs
            print 'prediction_class', pred_class, '\nprediction', pred
            #counter.update(predict(data, label, inputs,prediction, mode))
            if network_mode == 'class':
                right += check_pred(label, pred)
                show_stats(right, counter_runs)
    else:
        raise Exception("invalid mode")

def check_pred(label, pred):
    #pred =list(pred)
    r =  0
    #print pred
    print 'label', label
    for i in range(BATCH_SIZE):
        #print 'pred[i]', pred[i]
        #print 'label[i][pred[i]]',i,  label[i][pred[i]]
        #print 'label[i]', label[i]
        if label[i][pred[i]] == 1:
            r += 1
    return r

def show_stats(right, total):
    total = total *  16
    print 'Right prediction:' ,right, ",", 'from:' ,total
    print "In percent:", float(right)/ total


def stopwatch(value):
    valueD = (((value / 24) / 60) / 60)
    Days = int(valueD)

    valueH = (valueD - Days) * 24
    Hours = int(valueH)

    valueM = (valueH - Hours) * 60
    Minutes = int(valueM)

    valueS = (valueM - Minutes) * 60
    Seconds = int(valueS)

    print Days, ":", Hours, ":", Minutes, ":", Seconds


if __name__ == "__main__":

    start = time.time()
    # mode = parse_flags()
    print "start"
    if network_shape != 'regular':
        network_builder = build_cnn_model_inception
    else:
        network_builder = build_cnn_model

    if mode == "train":
        inputs, target_label, cost, optimizer, prediction, prediction_class = network_builder(mode)
        print "Train Dataset"
        data_set = create_dataset(TRAIN_DATA_SET, mode)
        print "END OF CREATE DATA SET"
        with tf.Session() as sess:
            run_session(data_set, cost, optimizer, prediction,prediction_class, inputs, target_label, mode, EPOCHS)
    else:
        inputs, target_label, prediction, prediction_class = network_builder(mode)
        print "Test Dataset"
        data_set = create_dataset(TEST_DATA_SET, mode)
        with tf.Session() as sess:
            run_session(data_set, [], [], prediction, prediction_class, inputs, target_label, mode, EPOCHS)

    end = time.time()
    stopwatch(end - start)

