from typing import List

import matplotlib.pyplot as plt


def plot_list(data: List, title):
    plt.plot(data)
    plt.title(title)
    plt.show()
