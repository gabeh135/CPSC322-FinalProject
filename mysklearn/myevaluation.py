"""
Programmer: Gabe Hoing
Class: CPSC 322, Fall 2024
Programming Assignment #7
11/20/2024
I attempted all bonus questions.

Description: This program creates Python representations for various
machine learning algorithm evaluation methods.
"""
import numpy as np # use numpy's random number generation
from tabulate import tabulate
from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    np.random.seed(random_state)
    n_samples = len(X)
    X_copy, y_copy = X[:], y[:]
    if shuffle:
        myutils.randomize_in_place(X_copy, y_copy)

    split = int(n_samples * (1 - test_size)) if test_size < 1 else int(n_samples - test_size)
    return X_copy[:split], X_copy[split:], y_copy[:split], y_copy[split:]

def stratified_train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into stratified train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    np.random.seed(random_state)
    X_copy, y_copy = X[:], y[:]

    class_indexes = {}

    for index, label in enumerate(y):
        if label not in class_indexes:
            class_indexes[label] = []
        class_indexes[label].append(index)
    
    train_indexes, test_indexes = [], []
    for label, indexes in class_indexes.items():
        if shuffle:
            myutils.randomize_in_place(indexes)
        split = int(len(indexes) * (1 - test_size)) if test_size < 1 else int(len(indexes) - test_size)
        train_indexes.extend(indexes[:split])
        test_indexes.extend(indexes[split:])

    train_indexes.sort()
    test_indexes.sort()

    X_train = [X_copy[i] for i in train_indexes]
    X_test = [X_copy[i] for i in test_indexes]
    y_train = [y_copy[i] for i in train_indexes]
    y_test = [y_copy[i] for i in test_indexes]

    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    np.random.seed(random_state)
    n = len(X)
    indices = list(range(n))

    if shuffle:
        myutils.randomize_in_place(indices)

    folds = []
    index = 0
    for k in range(n_splits):
        size = n // n_splits + 1 if k < n % n_splits else n // n_splits
        test_indices = indices[index:index + size]
        train_indices = [indices[i] for i in indices if i not in test_indices]

        folds.append((train_indices, test_indices))
        index += size

    return folds

# # BONUS function
# def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
#     """Split dataset into stratified cross validation folds.

#     Args:
#         X(list of list of obj): The list of instances (samples).
#             The shape of X is (n_samples, n_features)
#         y(list of obj): The target y values (parallel to X).
#             The shape of y is n_samples
#         n_splits(int): Number of folds.
#         random_state(int): integer used for seeding a random number generator for reproducible results
#         shuffle(bool): whether or not to randomize the order of the instances before creating folds

#     Returns:
#         folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
#             The first item in the tuple is the list of training set indices for the fold
#             The second item in the tuple is the list of testing set indices for the fold

#     Notes:
#         Loosely based on sklearn's StratifiedKFold split():
#             https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
#     """
#     np.random.seed(random_state)
#     _ = X

#     labels = {}
#     for index, label, in enumerate(y):
#         if label not in labels:
#             labels[label] = []
#         labels[label].append(index)

#     if shuffle:
#         for indices in labels.values():
#             myutils.randomize_in_place(indices)

#     splits = [[] for _ in range(n_splits)]
#     for indices in labels.values():
#         for i, index, in enumerate(indices):
#             splits[i % n_splits].append(index)

#     folds = []
#     for k in range(n_splits):
#         test_indices = splits[k]
#         train_indices = [index for fold in splits if fold != splits[k] for index in fold]
#         folds.append((train_indices, test_indices))
#     return folds

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    y_copy = y[:]
    y_enum = list(enumerate(y_copy))
    if shuffle is True:
        y_enum, _ = myutils.shuffle(y_enum, random_state=random_state)
    classes_grouped = []
    index_grouped = []
    for label in y_copy:
        temp = []
        if label not in classes_grouped:
            for i, value in y_enum:
                if value == label:
                    temp.append(i)
            classes_grouped.append(label)
            index_grouped.append(temp)

    instances = list(range(len(y)))
    # get folds
    folds = [([], []) for _ in range(n_splits)]
    fold_ind = 0
    for group in index_grouped:
        for value in group:
            folds[fold_ind][1].append(value)
            if fold_ind < n_splits - 1:
                fold_ind +=1
            else:
                fold_ind = 0
    # get train data
    for data in folds:
        test = data[1]
        for x in instances:
            if x not in test:
                data[0].append(x)
    return folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    np.random.seed(random_state)
    n = len(X)

    if not n_samples:
        n_samples = n

    indices = np.random.randint(0, n, n_samples)

    X_sample = [X[i] for i in indices]
    X_out_of_bag = [X[i] for i in range(n) if i not in indices]

    y_sample = [y[i] for i in indices]
    y_out_of_bag = [y[i] for i in range(n) if i not in indices]

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [[0] * len(labels) for _ in range(len(labels))]

    label_indices = {label: index for index, label in enumerate(labels)}

    for true, pred in zip(y_true, y_pred):
        matrix[label_indices[true]][label_indices[pred]] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    count = 0
    for true, pred in zip(y_true, y_pred):
        count += true == pred

    return count / len(y_pred) if normalize else count

def random_subsample(X, y, classifier, discretizer=None, k=10, test_size=0.33, random_state=None):
    """Uses random subsampling to compute classifier accuracy.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        classifier(method): Classifier being evaluated
        discretizer(method): Method used to discretize classifier results
        k(int): Times to run the train_test_split
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        mean_accuracy(float): average accuracy over k train test splits
        mean_error(float): average error over k train test splits
    """
    accuracies = []

    for _ in range(k):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        if discretizer:
            y_test = [discretizer(y) for y in y_test]
            y_pred = [discretizer(y) for y in y_pred]

        accuracies.append(accuracy_score(y_test, y_pred))

    mean_accuracy = sum(accuracies) / k

    return mean_accuracy, 1 - mean_accuracy

def cross_val_predict(X, y, classifier, discretizer=None, n_splits=5, random_state=None, shuffle=False, stratify=False):
    """Uses random subsampling to compute classifier accuracy.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        classifier(method): Classifier being evaluated
        discretizer(method): Method used to discretize classifier results
        n_splits(int): Number of kfold splits
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
        stratify(bool): whether to use stratification

    Returns:
        actual_values(list of obj): actual values in dataset
        pred_values(list of obj): parallel list to actual_values of predicted values
        accuracy(float): average accuracy over k train test splits
        error(float): average error over k train test splits
    """
    actual_values = []
    pred_values = []

    if stratify:
        folds = stratified_kfold_split(X, y, n_splits, random_state, shuffle)
    else:
        folds = kfold_split(X, n_splits, random_state, shuffle)

    for train_indices, test_indices in folds:
        X_train, X_test = [X[i] for i in train_indices], [X[i] for i in test_indices]
        y_train, y_test = [y[i] for i in train_indices], [y[i] for i in test_indices]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        if discretizer:
            y_test = [discretizer(y) for y in y_test]
            y_pred = [discretizer(y) for y in y_pred]

        actual_values.extend(y_test)
        pred_values.extend(y_pred)

    accuracy = accuracy_score(actual_values, pred_values)

    return actual_values, pred_values, accuracy, 1 - accuracy

def bootstrap_method(classifier, X, y=None, discretizer=None, k = 10, random_state=None):
    """Uses bootstrap sampling to compute classifier accuracy.

    Args:
        classifier(method): Classifier being evaluated
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        discretizer(method): Method used to discretize classifier results
        k(int): Times to run the train_test_split
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        mean_accuracy(float): average accuracy over k train test splits
        mean_error(float): average error over k train test splits
    """
    accuracies = []

    for _ in range(k):
        X_train, X_test, y_train, y_test = bootstrap_sample(X, y, random_state)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        if discretizer:
            y_test = [discretizer(y) for y in y_test]
            y_pred = [discretizer(y) for y in y_pred]

        accuracies.append(accuracy_score(y_test, y_pred))

    mean_accuracy = sum(accuracies) / k

    return mean_accuracy, 1 - mean_accuracy

def tabulate_confusion_matrix(matrix, headers):
    """Uses tabulate to create a readable version of a confusion matrix

    Args:
        matrix(list of list of obj): the confusion matrix being evaluated
        headers(list): headers for tabulate

    Returns:
        list of list of obj(the formatted confusion matrix)
    """
    table = []
    for i, row in enumerate(matrix):
        index = i + 1
        total = sum(row)
        recognition = (row[i] / total) if total > 0 else 0

        new_row = [headers[index]]
        new_row.extend(row)
        new_row.extend([total, int(recognition * 100)])
        table.append(new_row)

    return tabulate(table, headers=headers)

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    true_p, false_p, _, _ = accuracy_metrics(y_true, y_pred, labels, pos_label)

    return true_p / (true_p + false_p) if true_p + false_p > 0 else 0

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    true_p, _, _, false_n = accuracy_metrics(y_true, y_pred, labels, pos_label)
    return true_p / (true_p + false_n) if true_p + false_n > 0 else 0

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

def accuracy_metrics(y_true, y_pred, labels=None, pos_label=None):
    """Helper function to calculate True Positive, False Positive,
    True Negative, and False Negative counts.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        tuple: (tp, fp, tn, fn) the accuracy metrics of the prediction
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    true_p = false_p = true_n = false_n = 0

    for true, pred in zip(y_true, y_pred):
        if true not in labels or pred not in labels:
            continue

        if true == pos_label and pred == pos_label:
            true_p += 1
        elif true != pos_label and pred == pos_label:
            false_p += 1
        elif pos_label not in (true, pred):
            true_n += 1
        elif true == pos_label and pred != pos_label:
            false_n += 1

    return true_p, false_p, true_n, false_n

def display_cross_val(X, y, clf, labels, pos_label, matrix_headers, discretizer=None, n_splits=10, shuffle=True, stratify=True):
    """
    Computes and displays accuracy metrics for the given classifier using k-fold cross validation

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        classifier(method): Classifier being evaluated
        discretizer(method): Method used to discretize classifier results
        n_splits(int): Number of kfold splits
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
        stratify(bool): whether to use stratification

    Returns:
        actual_values(list of obj): actual values in dataset
        pred_values(list of obj): parallel list to actual_values of predicted values
        accuracy(float): average accuracy over k train test splits
        error(float): average error over k train test splits
    """
    actual, pred, accuracy, error = cross_val_predict\
        (X, y, clf, discretizer=discretizer, n_splits=n_splits, shuffle=shuffle, stratify=stratify)

    precision = binary_precision_score(actual, pred, labels, pos_label)
    recall = binary_recall_score(actual, pred, labels, pos_label)
    f1 = binary_f1_score(actual, pred, labels, pos_label)

    print(f"Accuracy: {accuracy}, Error Rate: {error}")
    print()

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    print()

    matrix = confusion_matrix(actual, pred, matrix_headers[1:3])
    print(tabulate_confusion_matrix(matrix, headers=matrix_headers))
