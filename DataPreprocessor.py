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
            label.append(data[-1])
        return feature, label

