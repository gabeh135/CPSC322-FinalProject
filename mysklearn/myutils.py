"""
Programmer: Gabe Hoing
Class: CPSC 322, Fall 2024
Programming Assignment #7
11/20/2024
I attempted all bonus questions.

Description: This program contains various utility functions
for mysklearn.
"""
import numpy as np

def randomize_in_place(alist, parallel_list=None):
    """Randomizes given list(s) in place.
    Args:
        alist(list): given list
        parallel_list(list): list parallel to alist if given
    """
    for i, _ in enumerate(alist):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def compute_euclidean_distance(v1, v2):
    """Computes the euclidean distance between
    the points of two lists.
    Args:
        v1(list of numeric vals): the first list of points
        v2(list of numeric vals): the second list of points
    Returns:
        int: the euclidean distance between the lists
    """
    return np.sqrt(sum(((v1[i] - v2[i]) ** 2 for i in range(len(v1)))))

def compute_categorical_distance(v1, v2):
    """Computes the distance between two categorical lists.
    Args:
        v1(list of categorical vals): the first list of points
        v2(list ofcategorical vals): the second list of points
    Returns:
        int: the distance between the lists
    """
    return np.sqrt(sum((val1 != val2) for val1, val2 in zip(v1, v2)))

def get_frequencies(values):
    """Returns the frequencies of values in a list.
    Args:
        values(list): list of values to compute
    Returns:
        list: unique values in given list
        list: parallel list containing counts of values
    """
    unique_values = sorted(list(set(values)))
    counts = [values.count(val) for val in unique_values]

    return unique_values, counts

def get_most_frequent(values):
    """Returns the the most frequent value in a list.
    Args:
        values(list): list of lists to compute
    Returns:
        str: the most frequent value
    """
    unique_values, counts = get_frequencies(values)
    return unique_values[counts.index(max(counts))]

def all_same_class(instances):
    """Determines whether the given instances are all of the same class
    Args:
        instances(list of list): list of instances
    Returns:
        bool: whether the instances all have the same class
    """
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False

    return True

def compute_partition_entropies(instances, partitions):
    """Computes the entropies for the given partitions
    Args:
        instances(list of list): list of instances
        partitions(dictionary of list of lists): the partitions to compute
    Returns:
        list: the entropies of the instances
    """
    entropies = []
    for partition in partitions.values():
        partition_entropy = 0
        values = [value[-1] for value in partition]
        for value in set(values):
            partition_entropy -= (values.count(value) / len(partition)) * np.log2(values.count(value) / len(partition))
        entropies.append((len(partition) / len(instances)) * partition_entropy)

    return entropies

def shuffle(x, y=None, random_state=None):
    """Shuffles data for randomized sampling

    Args:
        x(list): list in need of shuffling
        y(list): list if wanting to shuffle in parallel to x
        random_state(int): seed for np.random
    
    Returns:
        randomized_x(list): x after randomized shuffling
        randomized_y(list): y after randomized shuffling, 
            parallel to randomized_X
    """
    if random_state is not None:
        np.random.seed(random_state)
    randomized_x = x[:]
    if y is not None:
        randomized_y = y[:]
    else:
        randomized_y = None
    for i in range(len(x)):
        # pick an index to swap
        j = np.random.randint(0, len(x)) # random int in [0,n)
        randomized_x[i], randomized_x[j] = randomized_x[j], randomized_x[i]
        if y is not None:
            randomized_y[i], randomized_y[j] = randomized_y[j], randomized_y[i]
    return randomized_x, randomized_y