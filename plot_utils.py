"""
Programmer: Sean Cowan

Class: CPSC 322, Fall 2024

Programming Assignment #3

Description: Contains useful plotting functions
"""
import matplotlib.pyplot as plt
from mysklearn import myutils

def bar_chart(x , y, title, xlabel, ylabel, xtick_labels=None):
    plt.figure()
    plt.bar(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xtick_labels != None:
        plt.xticks(x, xtick_labels, rotation=15)
    plt.show()

def histogram(data, title, xlabel, ylabel):
    plt.figure()
    plt.hist(data, bins=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def linear_regression(x, y, m, b, title, xlabel, ylabel):
    plt.figure() 
    plt.scatter(x, y, marker="x")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c='red', lw=3)
    r = myutils.compute_r(x, y)
    ax = plt.gca()
    ax.annotate("r=%.2f" %r, xy=(0, 1), xycoords='axes fraction',
                color="r", bbox=dict(boxstyle="round", fc="1", color="r"))
    plt.show()

def scatter_chart(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.scatter(x, y, marker="x")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def box_plot(distributions, tick_labels=None):
    plt.figure()
    plt.boxplot(distributions, vert=False, labels=tick_labels)
    plt.show()