#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:34:01 2018

@author: csn
"""
import os
import random
import numpy as np
import pickle

num_classes = 14


class InputData(object):
    def __init__(self, folder_name, shuffle=True):
        """
        Note: Existing non data files in the folder will raise an exception
        :param folder_name: The name of folder only including data files
        :param shuffle: Whether shuffle data in each files or not
        """
        self.files_list = [''.join([folder_name, '/', i]) for i in os.listdir(folder_name)]
        self.num_file = len(self.files_list)
        self.shuffle = shuffle
        if shuffle:
            self.order_files = random.sample(list(range(self.num_file)), self.num_file)
            self.files_list = [self.files_list[i] for i in self.order_files]
        else:
            self.order_files = list(range(self.num_file))
        self.current_file_index = 0
        self.current_video_index = 0
        with open(self.files_list[0], 'rb') as f:
            self.data = pickle.load(f)
            # print(self.files_list[self.current_file_index])  ##
        self.num_feature = len(self.data)
        if shuffle:
            self.order_feature = random.sample(list(range(self.num_feature)), self.num_feature)
            self.data = [self.data[i] for i in self.order_feature]
        else:
            self.order_feature = list(range(self.num_feature))

    def __check_index(self, size):
        if self.current_video_index + size <= self.num_feature:
            data = self.data[self.current_video_index: self.current_video_index+size]
            self.current_video_index += size
            return data
        else:
            num_excess = self.current_video_index + size - self.num_feature
            data1 = self.data[self.current_video_index: self.num_feature]
            self.current_file_index += 1
            if self.current_file_index == self.num_file:
                if self.shuffle:
                    self.order_files = random.sample(list(range(self.num_file)), self.num_file)
                    self.files_list = [self.files_list[i] for i in self.order_files]
                else:
                    self.order_files = list(range(self.num_file))
                self.current_file_index = 0
            with open(self.files_list[self.current_file_index], 'rb') as f:
                self.data = pickle.load(f)
            self.num_feature = len(self.data)
            if self.shuffle:
                self.order_feature = random.sample(list(range(self.num_feature)), self.num_feature)
                self.data = [self.data[i] for i in self.order_feature]
            else:
                self.order_feature = list(range(self.num_feature))
            data2 = self.data[0: num_excess]
            self.current_video_index = num_excess
            return data1 + data2
    
    def next_batch(self, size):
        data = self.__check_index(size)
        feature = []
        labels = []
        dims = []
        for i in range(size):
            # if data[i]['feature'].shape[0] > 400 and data[i]['feature'].ndim != 1:
            #     feat = data[i]['feature'][0: 400, :]
            #     feature.append(feat)
            # else:
            feature.append(data[i]['feature'])
            
            if data[i]['label'] == 0:
                labels.append([0.])
            else:
                labels.append([1.])
            dims.append(data[i]['num'])
        return feature, np.array(labels, dtype=np.float32), np.array(dims, dtype=np.float32)
