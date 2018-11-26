import pandas as pd
import numpy as np
import numbers
import json
import random
from random import shuffle


def prepare_data_vectors(data, start_data, vec, label, vec_size, batch_size,limit,network_mode):
    input_vectors = []
    input_label = []
    counter = 0
    for i in range(start_data, min(len(data["Open"]),limit+start_data)):
        k = 0
        for j in range(i - vec_size, i):
            if data["Open"][j] != '\xe2\x80\x94':
                vec[counter][0][k][0] = data["Open"][j]
            elif data["Open"][j] == '\xe2\x80\x94' and k==0:
                vec [counter][0][k][0] = vec[counter-1][0][vec_size-1][0]
            else:
                vec[counter][0][k][0] = vec[counter][0][k-1][0]
            k += 1
        if data["Open"][i] != '\xe2\x80\x94':
            target = data["Open"][i]
        else:
            target = vec[counter][0][vec_size-1][0]

        if network_mode == 'reg':
            label[counter] = target
        if network_mode == 'class':
            x = float(target) / vec[counter][0][vec_size - 1][0] - 1
            label[counter][inclass(x)] = 1

        #print 'vec[counter]', vec[counter]
        #print 'label[counter', label[counter]

        input_vectors.append(vec[counter])#counter - 1 \ need to be vec[counter] instead of y
        input_label.append(label[counter])
        counter += 1
        batch = np.array(input_vectors), np.array(input_label) #/ i change the place of this line

        if counter % batch_size == 0:

            yield batch
            input_vectors = []
            input_label = []


def clear_data(data, data_np):
    for i in range(len(data['Open'])):
        if data['Open'][i] != '\xe2\x80\x94':
            data_np = np.append(data_np, float(data['Open'][i]))
        else:
            j = i + 1
            while data['Open'][j] == '\xe2\x80\x94':
                j += 1
            data_np = np.append(data_np, float(data['Open'][j]))
    #for value in data_np:
    #    print value
    return data_np

def prepare_data_vectors_rand(data, start_data, vec, label, vec_size, limit, network_mode):
    #data_np = np.zeros(len(data['Open']), dtype= 'float64' )
    data_np = np.array([]).astype('float64')

    print "now"
    data_np = clear_data(data, data_np)

    data_np = (data_np - np.mean(data_np, axis = 0))
    data_np = data_np / np.std(data_np, axis=0)

    list_num = []
    input_vectors = []
    input_label = []
    counter = 0
    for i in range(start_data, min(len(data["Open"]),limit+start_data)):
        list_num.append(counter)
        k = 0
        for j in range(i - vec_size, i):
            vec[counter][0][k][0] = data_np[j]
            k += 1
        target = data_np[i]

        if network_mode == 'reg':
            label[counter] = target
        if network_mode == 'class':
            x = float(target)/vec[counter][0][vec_size-1][0]-1
            label[counter][inclass(x)] = 1

        input_vectors.append(vec[counter])#counter - 1 \ need to be vec[counter] instead of y
        input_label.append(label[counter])
        counter += 1

    return list_num,input_vectors,input_label



def get_data(batch_size, list_num, input_vectors, input_label):

    print "End of prepare data"

    shuffle(list_num)

    counter = 0
    vectors_to_send = []
    labels_to_send = []
    for num in list_num:
        vectors_to_send.append(input_vectors[num])
        labels_to_send.append(input_label[num])
        #batch = np.array(vectors_to_send), np.array(labels_to_send)
        counter += 1
        if counter % batch_size == 0:
            batch = np.array(vectors_to_send), np.array(labels_to_send)
            # print "END OF PREPARE DATA"
            yield batch
            # batch[:] = []
            vectors_to_send = []
            labels_to_send = []


def inclass(x):
    x_percent = 100 * x
    min_num = -50
    max_num = 50
    if x_percent < min_num: return 0
    if -50 <= x_percent < -30: return 1
    if -30 <= x_percent < -20: return 2
    if -20 <= x_percent < -15: return 3
    if -15 <= x_percent < -10: return 4
    if -10 <= x_percent < -7.5: return 5
    if -7.5 <= x_percent < -5: return 6
    if -5 <= x_percent < -2.5: return 7
    if -2.5 <= x_percent < 0: return 8
    if 0 <= x_percent < 2.5: return 9
    if 2.5 <= x_percent < 5: return 10
    if 5 <= x_percent < 7.5: return 11
    if 7.5 <= x_percent < 10: return 12
    if 10 <= x_percent < 15: return 13
    if 15 <= x_percent < 20: return 14
    if 20 <= x_percent < 30: return 15
    if 30 <= x_percent < 50: return 16
    if max_num <= x_percent: return 17


    '''
    x_percent = 100 * x
    min_num = -50
    max_num = 50
    jump_num = 2.5
    intervals = int((max_num-min_num)/jump_num)
    for i in range(intervals):
        if  min_num + jump_num*i <= x_percent < min_num +jump_num*(i+1):
            return  i
    if x_percent >= max_num: return intervals
    if x_percent < min_num: return intervals +1


def prepare_data_vectors_rand(data, start_data, vec, label, vec_size, limit, network_mode):
    data_np = np.zeros(len(data['Open']))
    data_np = data['Open']
    print "now"
    print data['Open']
    print 'data_norm'
    print data_np

    exit()
    list_num = []
    input_vectors = []
    input_label = []
    counter = 0
    for i in range(start_data, min(len(data["Open"]),limit+start_data)):
        list_num.append(counter)
        k = 0
        for j in range(i - vec_size, i):
            if data["Open"][j] != '\xe2\x80\x94':
                vec[counter][0][k][0] = data["Open"][j]
            elif data["Open"][j] == '\xe2\x80\x94' and k==0:
                vec [counter][0][k][0] = vec[counter-1][0][vec_size-1][0]
            else:
                vec[counter][0][k][0] = vec[counter][0][k-1][0]
            k += 1

        if data["Open"][i] != '\xe2\x80\x94':
            target = data["Open"][i]
        else:
            target = vec[counter][0][vec_size-1][0]

        if network_mode == 'reg':
            label[counter] = target
        if network_mode == 'class':
            x = float(target)/vec[counter][0][vec_size-1][0] - 1
            label[counter][inclass(x)] = 1

        input_vectors.append(vec[counter])#counter - 1 \ need to be vec[counter] instead of y
        input_label.append(label[counter])
        counter += 1
    return list_num,input_vectors,input_label



    '''