# pylint: skip-file
"""test_randomforest.py

@author gabeh135
"""

import numpy as np
from sklearn.model_selection import train_test_split

from mysklearn import myutils
from mysklearn import myevaluation
from mysklearn.myclassifiers import MyRandomForestClassifier, MyDecisionTreeClassifier

def test_compute_random_subset():
    # test normal functionality
    header = ["att0", "att1", "att2", "att3"]
    num_atts = 2 

    subset = myutils.compute_random_subset(header, num_atts)

    assert len(set(subset)) == 2

    # test num_atts > header size
    header = ["att0", "att1"]
    num_atts = 3

    subset = myutils.compute_random_subset(header, num_atts)

    assert len(set(subset)) == 2

def test_select_top_elements():
    elements = ["a", "b", "c", "d", "e"]
    values = [-10000, 0, -1.5, 1, 20]
    n_elements = 2

    selected_indexes = myutils.select_top_elements(elements, values, n_elements)

    assert selected_indexes == [3, 4]

    elements = ["a", "b", "c", "d", "e"]
    values = [5, 4, 3, 2, 1]

    selected_indexes = myutils.select_top_elements(elements, values, n_elements)

    assert selected_indexes == [0, 1]

def test_stratified_train_test_split():
    """
    Test the custom stratified_train_test_split against sklearn's train_test_split.
    """
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    X_train_solution, X_test_solution, y_train_solution, y_test_solution = \
        train_test_split(X_train_interview, y_train_interview, test_size=0.33, random_state=0, stratify=y_train_interview)
    X_train, X_test, y_train, y_test = \
        myevaluation.stratified_train_test_split(X_train_interview, y_train_interview, test_size=0.33, random_state=0, shuffle=True)
    
    for value in set(y_train_interview):
        assert X_train.count(value) == X_train_solution.count(value)
        assert X_test.count(value) == X_test_solution.count(value)
        assert y_train.count(value) == y_train_solution.count(value)
        assert y_test.count(value) == y_test_solution.count(value)

def test_discretize_ranking():
    ratings = [1311, 800, 972, 1000, 1200]
    ranks = [myutils.discretize_ranking(rating) for rating in ratings]

    assert ranks == ["very high", "very low", "below average", "average", "above average"]

def test_random_forest_classifier_fit():
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    N = 5
    M = 3
    F = 2

    interview_clf = MyRandomForestClassifier()
    interview_clf.fit(X_train_interview, y_train_interview, N, M, F)

    assert len(interview_clf.forest) == M

    assert interview_clf.forest[0].tree != interview_clf.forest[1].tree
    assert interview_clf.forest[0].tree != interview_clf.forest[2].tree
    assert interview_clf.forest[1].tree != interview_clf.forest[2].tree

def test_random_forest_classifier_predict():
    # create three decision tree classifiers using a random subsetting and the interview set
    # make a random forest equal to this

    # test that the majority of the three classifiers equals decision predict.
    assert False is True

X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]
y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
X_test_iphone = [[2, 2, "fair"], [1, 1, "excellent"]]

clf_1 = MyDecisionTreeClassifier()
clf_1.fit(X_train_interview, y_train_interview, 2)
pred_1 = clf_1.predict(X_test_iphone)
print(pred_1)

clf_2 = MyDecisionTreeClassifier()
clf_2.fit(X_train_interview, y_train_interview, 2)
pred_2 = clf_2.predict(X_test_iphone)
print(pred_2)

clf_3 = MyDecisionTreeClassifier()
clf_3.fit(X_train_interview, y_train_interview, 2)
pred_3 = clf_3.predict(X_test_iphone)
print(pred_3)

