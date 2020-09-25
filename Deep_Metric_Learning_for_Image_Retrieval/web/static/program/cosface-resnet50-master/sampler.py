import os
import random
import numpy as np
"""
This is the sampler for image pairs
@:param data_path : the path of the dataset
@:param out_path : the path of the output (~/out_path/pairs.txt)
@:param samples_per_label=0 : int, the number of samples for each label, half for 1 and half for 0
"""


def get_pair_sample_list(data_path, out_path, samples_per_label=0):
    label_path = os.listdir(data_path)
    list_length = len(label_path) * samples_per_label
    positive_num = int(samples_per_label / 2)
    negative_num = samples_per_label - positive_num
    print('Sampling {} samples'.format(list_length))
    positive_list = []
    negative_list = []
    for i in range(len(label_path)):
        temp_list = os.listdir(os.path.join(data_path, label_path[i]))
        p_selected = random.sample(temp_list, positive_num * 2)
        p_left = p_selected[:positive_num]
        p_right = p_selected[positive_num:]
        for j in range(positive_num):
            positive_list.append([os.path.join(label_path[i], p_left[j]), os.path.join(label_path[i], p_right[j]), 1])

        n_label_selected = []
        if i == 0:
            n_label_selected.extend(label_path[1:])
        elif i == len(label_path):
            n_label_selected.extend(label_path[:-1])
        else:
            n_label_selected.extend(label_path[:i])
            n_label_selected.extend(label_path[i+1:])

        n_left = random.sample(temp_list, positive_num)
        n_label = random.sample(n_label_selected, negative_num)
        n_right = []
        for j in range(negative_num):
            n_right.append(os.path.join(n_label[j], ','.join(random.sample(os.listdir(os.path.join(data_path, n_label[j])), 1))))

        for j in range(negative_num):
            negative_list.append([os.path.join(label_path[i], n_left[j]), n_right[j], 0])

    result = []
    result.extend(positive_list)
    result.extend(negative_list)
    #print(result)
    return result
    #result = np.array(result)
    #np.savetxt(os.path.join(out_path, 'pairs.txt'), result, fmt='%s')


#if __name__ == '__main__':
#    get_pair_sample_list('/mnt/batch/tasks/shared/LS_root/mounts/clusters/yikai/code/Users/yili.lai/recognition/CUB_200_2011/images', '/mnt/batch/tasks/shared/LS_root/mounts/clusters/yikai/code/Users/yikai.wang', 8)
