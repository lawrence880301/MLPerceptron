import random
import numpy as np
from MLPerceptron import *

class Datapreprocessor():
    def __init__(self) -> None:
        pass

    def readfile(url):
        read = open(url)
        file = read.readlines()
        read.close()
        return file
    def text_to_numlist(dataset):
        """load text dataset to numeracial list dataset

        Args:
            dataset (string): txt or other file

        Returns:
            dataset: float_list
        """
        dataset = [data.strip().split() for data in dataset]
        dataset = [list(map(float,data)) for data in dataset]
        return dataset
    def train_test_split(data,split_ratio):
        """shuffle data and spilt it into train and test data

        Args:
            data (list): numeracial list
            split_ratio (float): portion of train

        Returns:
            train_data, test_data: float_list
        """
        sample_position = round(len(data)*(split_ratio))
        random.shuffle(data)
        train_data, test_data = data[:sample_position], data[sample_position:]
        return train_data, test_data

    def feature_label_split(dataset):
        feature, label = [], []
        for data in dataset:
            feature.append(data[:-1])
            label.append([data[-1]])
        return feature, label

    def label_list(dataset):
        label_list = []
        for data in dataset:
            if data[-1] not in label_list:
                label_list.append(data[-1])
        return label_list

    def num_of_feature(dataset):
        return len(dataset[0][:-1])
    
    def label_preprocess(labelset):
        modified_label_set = []
        existing_label = {}
        new_max_label = 0
        for label in labelset:
            if str(label) not in existing_label:
                existing_label[str(label)] = new_max_label
                modified_label_set.append([existing_label[str(label)]])
                new_max_label+=1
            else:
                modified_label_set.append([existing_label[str(label)]])
        return modified_label_set


