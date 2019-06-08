from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_l2_norm(data, x):
    plt.ylim(1e10, 1e13)
    plt.xlabel('size of training set')
    plt.plot(x, data, marker='o')
    plt.title('L2 Norm')
    plt.show()


def plot_spectral(data: List, x):
    plt.ylim(1e2, 1e7)
    plt.xlabel('size of training set')
    plt.plot(x, data, marker='o')
    plt.title('Spectral Norm')
    plt.show()


def plot_l2_path(data: List, x):
    plt.ylim(0, 200)
    plt.xlabel('size of training set')
    plt.plot(x, data, marker='o')
    plt.title('L2 Path Norm')
    plt.show()


def plot_l1_path(data: List, x):
    plt.ylim(1e10, 1e16)
    plt.xlabel('size of training set')
    plt.plot(x, data, marker='o')
    plt.title('L1 Path Norm')
    plt.show()


def plot_error(data1: List, data2: List, x, title):
    # data1.insert(4, 0.02)
    # data2.insert(4, 0.89)
    # data1.insert(8, 0.2)
    # data2.insert(8, 0.9)
    plt.plot(x, data1, marker='o')
    plt.plot(x, data2, marker='o')
    plt.ylabel('error')
    plt.xlabel('size of training set')
    plt.title(title)
    plt.show()


def plot_all(data, x, title):
    l2, l1_path, l2_path, spectral = data
    # l2.insert(4, 1.044356499327962e-09)
    # spectral.insert(4, 1.0256027650833768e-09)
    # l1_path.insert(4, 9.820030170477851)
    #l2_path.insert(4, 0.7)
    l2 = np.asarray(l2)
    l2_path = np.asarray(l2_path)
    l2 = (l2 - np.min(l2)) / np.ptp(l2)
    l2_path = (l2_path - np.min(l2_path)) / np.ptp(l2_path)
    l1_path = (l1_path - np.min(l1_path)) / np.ptp(l1_path)
    spectral = (spectral - np.min(spectral)) / np.ptp(spectral)
    l2[0] = 0.01
    l1_path[0] = 0.01
    l2_path[0] = 0.01
    spectral[0] = 0.01
    y_l2, = plt.plot(x, l2, marker='o', label='L2 Norm')
    y_l1path, = plt.plot(x, l1_path, marker='o', label='L1-Path-Norm')
    y_l2path, = plt.plot(x, l2_path, marker='o', label='L2-Path-Norm')
    y_spectral, = plt.plot(x, spectral, marker='o', label='Spectral')
    plt.legend([y_l2, y_l1path, y_l2path, y_spectral])
    plt.ylabel('error')
    plt.xlabel('size of training set')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_list(list, title):
    plt.plot(list)
    plt.title(title)
    plt.show()