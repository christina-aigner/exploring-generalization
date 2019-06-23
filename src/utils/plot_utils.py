from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_error(data1: List, data2: List, x, xlabel, title):
    """
    Plots the error functions for a given model.
    Args:
        data1: train error
        data2: test error
        x: the model names on the x axis
        xlabel: label of the x axis
        title: title for the plot

    """
    plt.rcParams.update({'font.size': 16})
    train, = plt.plot(x, data1, marker='o', label='train')
    test, = plt.plot(x, data2, marker='o', label='test')
    plt.ylabel('error')
    plt.xlabel(xlabel)
    plt.legend([train, test])
    plt.legend()
    plt.title(title)
    plt.show()


def plot_normalized(data, x, xlabel, title):
    """
    Plots measures in normalized form within a range of 0 and 1.
    Args:
        data: 4-Tuple of calculated norms
        x: the model names
        xlabel: label of the x axis
        title: title of the plot

    Returns:

    """
    plt.rcParams.update({'font.size': 16})
    l2, l1_path, l2_path, spectral = data
    l2 = np.asarray(l2)
    l2_path = np.asarray(l2_path)
    # normalization
    l2 = (l2 - np.min(l2)) / np.ptp(l2)
    l2_path = (l2_path - np.min(l2_path)) / np.ptp(l2_path)
    l1_path = (l1_path - np.min(l1_path)) / np.ptp(l1_path)
    spectral = (spectral - np.min(spectral)) / np.ptp(spectral)
    y_l2, = plt.plot(x, l2, marker='P', label='L2 Norm', color='r')
    y_l1path, = plt.plot(x, l1_path, marker='s', label='L1-Path-Norm', color='b')
    y_l2path, = plt.plot(x, l2_path, marker='D', label='L2-Path-Norm', color='m')
    y_spectral, = plt.plot(x, spectral, marker='X', label='Spectral', color='g')
    plt.legend([y_l2, y_l1path, y_l2path, y_spectral])
    plt.ylabel('error')
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_list(list, title, x):
    """ Just makes a standard 2 dimensional plot. """
    plt.plot(x, list)
    plt.title(title)
    plt.show()
