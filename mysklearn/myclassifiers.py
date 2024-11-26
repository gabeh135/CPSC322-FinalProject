"""
Programmer: Gabe Hoing
Class: CPSC 322, Fall 2024
Programming Assignment #7
11/20/2024
I attempted all bonus questions.

Description: This program creates Python representations for various
machine learning classifiers.
"""
import operator
import os
from mysklearn import myevaluation
from mysklearn import myutils

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3, categorical=False):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
            categorical(bool): whether computing categorical values
        """
        self.n_neighbors = n_neighbors
        self.categorical = categorical
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric or categorical vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for row1 in X_test:
            row_indexes_dists = []
            for j, row2 in enumerate(self.X_train):
                if self.categorical:
                    dist = myutils.compute_categorical_distance(row1, row2)
                else:
                    dist = myutils.compute_euclidean_distance(row1, row2)
                row_indexes_dists.append((j, dist))
            row_indexes_dists.sort(key=operator.itemgetter(-1))
            top_k = row_indexes_dists[:self.n_neighbors]

            distances.append([row[1] for row in top_k])
            neighbor_indices.append([row[0] for row in top_k])

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric or categorical vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, neighbor_indices = self.kneighbors(X_test)

        y_predicted = []
        for indices in neighbor_indices:
            y_predicted.append(myutils.get_most_frequent([self.y_train[index] for index in indices]))

        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        _ = X_train
        self.most_common_label = myutils.get_most_frequent(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label for _ in X_test]

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.priors = {}
        self.posteriors = {}
        y_counts = {}

        for label in y_train:
            y_counts[label] = y_counts.get(label, 0) + 1

        length = len(y_train)
        for label, count in y_counts.items():
            self.priors[label] = count / length

        for label in y_counts:
            self.posteriors[label] = [{} for _ in range(len(X_train[0]))]

        for i, row in enumerate(X_train):
            label = y_train[i]
            for j, value in enumerate(row):
                if value not in self.posteriors[label][j]:
                    self.posteriors[label][j][value] = 0
                self.posteriors[label][j][value] += 1

        for label, counts in self.posteriors.items():
            label_count = y_counts[label]
            for value_counts in counts:
                for value, count in value_counts.items():
                    value_counts[value] = count / label_count

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        probabilities = {}

        for row in X_test:
            for label in self.priors:
                probability = self.priors[label]

                for index, value in enumerate(row):
                    counts = self.posteriors[label][index]
                    if value in counts:
                        probability *= counts[value]

                probabilities[label] = probability

            y_pred.append(max(probabilities, key=lambda key: probabilities[key]))

        return y_pred

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        header(list of str): The attribute header
        attribute_domains(dictionary): Which attributes are in the "domain" of each header item.
        tree(nested list): The extracted tree model.
        subset_size(int): The random subset size, if applicable.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.header = None
        self.attribute_domains = None
        self.tree = None
        self.subset_size = None

    def fit(self, X_train, y_train, subset_size=None):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
            subset_size(int): If specified, defines the number of attributes to randomly select (F)

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # one of the first things I should do is programatically create attribute domains
        self.subset_size = subset_size
        self.header = [f"att{i}" for i in range(len(X_train[0]))]
        self.attribute_domains = {self.header[index]: sorted({row[index] for row in X_train}) for index in range(len(self.header))}

        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = self.header.copy()
        self.tree = self.tdidt(train, available_attributes)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.tdidt_predict(self.tree, instance) for instance in X_test]

    def select_attribute(self, instances, attributes):
        """Selects attribute with the lowest entropy

        Args:
            instances(list of list): list of instances containing attributes
            attributes(list): list of objects available to choose from
            subset_size(int or None): The size of the random subset of attributes to consider.

        Returns:
            str: the chosen attribute
        """
        atts = attributes[:]
        if self.subset_size:
            atts = myutils.compute_random_subset(atts, self.subset_size)

        entropies = []
        for attribute in atts:
            partitions = self.partition_instances(instances, attribute)
            entropies.append(sum(myutils.compute_partition_entropies(instances, partitions)))

        return atts[entropies.index(min(entropies))]

    def partition_instances(self, instances, attribute):
        """Partitions the instances by the given attribute

        Args:
            instances(list of list): list of instances containing attributes
            attribute: the attribute to partition by

        Returns:
            dictionary of list of lists: the partitioned instances
        """
        att_index = self.header.index(attribute)
        att_domain = self.attribute_domains[attribute]
        partitions = {}
        for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def tdidt(self, current_instances, available_attributes):
        """Generates decision tree using the tdidt method

        Args:
            current_instances(list of list): list of instances containing attributes and a class
            available_attributes(list): list of attributes not yet used in the given branch

        Returns:
            tree: the generated decision tree of nested lists
        """
        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, available_attributes)
        available_attributes.remove(split_attribute)
        tree = ["Attribute", split_attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        partition_sum = sum(len(partition) for partition in partitions.values())
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value in sorted(partitions.keys()): # process in alphabetical order
            att_partition = partitions[att_value]
            value_subtree = ["Value", att_value]
            #    CASE 1: all class labels of the partition are the same
            # => make a leaf node
            if len(att_partition) > 0 and myutils.all_same_class(att_partition):
                leaf = ["Leaf", att_partition[0][-1], len(att_partition), partition_sum]
                value_subtree.append(leaf)
                tree.append(value_subtree)
            #    CASE 2: no more attributes to select (clash)
            # => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                atts = [att[-1] for att in att_partition]
                max_att = max(atts, key=atts.count)
                leaf = ["Leaf", max_att, atts.count(max_att), partition_sum]
                value_subtree.append(leaf)
                tree.append(value_subtree)
            #    CASE 3: no more instances to partition (empty partition)
            # => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                atts = [instance[-1] for instance in current_instances]
                max_att = max(atts, key=atts.count)
                tree = ["Leaf", max_att, atts.count(max_att), len(current_instances)]
                break
            else:
                # none of base cases were true, recurse!!
                subtree = self.tdidt(att_partition, available_attributes.copy())
                value_subtree.append(subtree)
                tree.append(value_subtree)
        return tree

    def tdidt_predict(self, tree, instance):
        """Recursively traverses the tree in order to generate a prediction

        Args:
            tree(nested list of str): the tree or subtree being traversed
            instance(list): the instance to generate a prediction of

        Returns:
            tree: the generated decision tree of nested lists
        """
        info_type = tree[0]
        if info_type == "Leaf":
            return tree[1]

        att_index = self.header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance[att_index]:
                return self.tdidt_predict(value_list[2], instance)
        return ""

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attribute_names = self.header

        def print_rules(tree, prefix=""):
            """Helper function to recursively print the decision rules.

            Args:
                tree(nested list of str): The tree being printed
            """
            if tree[0] == "Leaf":
                class_label = tree[1]
                print(f"{prefix}THEN {class_name} = {class_label}")
            else:
                attribute = tree[1]
                att_index = self.header.index(attribute)
                att_name = attribute_names[att_index]
                for i in range(2, len(tree)):
                    value_list = tree[i]
                    att_value = value_list[1]
                    new_prefix = f"{prefix}{"AND" if prefix else "IF"} {att_name} == {att_value} "
                    print_rules(value_list[2], new_prefix)

        print_rules(self.tree)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """

        if attribute_names is None:
            attribute_names = self.header

        label_counts = {}

        def visualize_helper(tree, label_counts):
            """Helper function to recursively create a Graphviz graph visualization.

            Args:
                tree(nested list of str): The tree being printed

            Returns:
                string: The Graphviz visualization
                string: The identifier of the tree's root node
            """
            info_type = tree[0]
            class_label = tree[1]

            label_counts[class_label] = label_counts.get(class_label, 0) + 1
            class_id = f'{class_label}_{label_counts[class_label]}'

            if info_type == "Leaf":
                return f'    {class_id}[label="{class_label}"];\n', class_id

            att_index = self.header.index(class_label)
            att_name = attribute_names[att_index]
            label_string = f'    {class_id}[label="{att_name}"];\n'

            for i in range(2, len(tree)):
                value_list = tree[i]
                att_value = value_list[1]
                child, child_id = visualize_helper(value_list[2], label_counts)
                label_string += child
                label_string += f'    {class_id} -- {child_id}[label="{att_value}"];\n'
            return label_string, class_id

        graph = "graph g {\n"
        tree_graph, _ = visualize_helper(self.tree, label_counts)
        graph += tree_graph
        graph += "}\n"

        with open(dot_fname, "w", encoding="utf-8") as outfile:
            outfile.write(graph)

        os.popen(f"dot -Tpdf -o {pdf_fname} {dot_fname}")

class MyRandomForestClassifier: 
    """Represents a random forest classifier.

    Attributes:
        forest(list of decision trees): The generated decision trees.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def init(self):
        self.forest = None

    def fit(self, X_train, y_train, n_samples, m_classifiers, subset_size):
        # Generate a random stratified test set.
        folds = myevaluation.stratified_kfold_split(X_train, y_train, n_splits=3, random_state=0, shuffle=True)
        train_indexes, test_indexes = folds[0]

        X_remainder, X_test = [X_train[i] for i in range(len(X_train)) if i in train_indexes], [X_train[i] for i in range(len(X_train)) if i in test_indexes]
        y_remainder, y_test = [y_train[i] for i in range(len(y_train)) if i in train_indexes], [y_train[i] for i in range(len(y_train)) if i in test_indexes]
        # X_remainder, X_test, y_remainder, y_test = myevaluation.stratified_train_test_split(X_train, y_train, test_size=0.33, random_state=0, shuffle=True)

        # Generate N decision trees using bootstrapping over the remainder set.
        trees = []
        accuracies = []
        for _ in range(n_samples):
            X_train, X_validate, y_train, y_validate = myevaluation.bootstrap_sample(X_remainder, y_remainder)

            clf = MyDecisionTreeClassifier()

            # At each node, decision trees by randomly selecting F of the remaining attributes as candidates to partition on.
            clf.fit(X_train, y_train, subset_size=subset_size)
            y_pred = clf.predict(X_validate)
            accuracy = myevaluation.accuracy_score(y_validate, y_pred)
            trees.append(clf)
            accuracies.append(accuracy)

        # Select the M most accurate of the N decision trees using the corresponding validation sets.
        top_indexes = myutils.select_top_elements(trees, accuracies, m_classifiers)
        self.forest = [trees[i] for i in top_indexes]

        # Return the test set for future evaluation
        return X_test, y_test

    def predict(self, X_test):
        """Makes predictions for test instances in X_test using majority voting.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_pred(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = [tree.predict(X_test) for tree in self.forest]

        # Use simple majority voting to predict classes using the M decision trees over the test set.
        y_pred = []
        for index in range(len(X_test)):
            X_predictions = [predictions[i][index] for i in range(len(predictions))]
            y_pred.append(myutils.get_most_frequent(X_predictions))
        
        return y_pred
