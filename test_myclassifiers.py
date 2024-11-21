# pylint: skip-file
"""test_myclassifiers.py

@author gabeh135
Note: do not modify this file
"""
import numpy as np

from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier

def discretize_regression_classifier_test(y_value):
    return "high" if y_value >= 100 else "low"

def test_kneighbors_classifier_kneighbors():
    X_train = [[1, 1], [1, 0], [1 / 3, 0], [0, 0]]
    y_train = ["bad", "bad", "good", "good"]
    unseen_instance = [1 / 3, 1]

    clf = MyKNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    distances, indexes = clf.kneighbors([unseen_instance])

    assert np.allclose(distances, [0.6666666666666666, 1.0, 1.0540925533894598])
    assert np.allclose(indexes, [0, 2, 3])

    X_train = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    unseen_instance = [2, 3]

    clf = MyKNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    distances, indexes = clf.kneighbors([unseen_instance])

    assert np.allclose(distances, [1.41421356, 1.41421356, 2.0])
    assert np.allclose(indexes, [0, 4, 6])

    X_train = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]
    y_train = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",
            "-", "-", "+", "+", "+", "-", "+"]
    unseen_instance = [9.1, 11.0]

    clf = MyKNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)

    distances, indexes = clf.kneighbors([unseen_instance])

    assert np.allclose(distances, [0.6082762530298216, 1.2369316876852974, 2.202271554554525, 2.8017851452243794, 2.9154759474226513])
    assert np.allclose(indexes, [6, 5, 7, 4, 8])

def test_kneighbors_classifier_predict():
    X_train = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train = ["bad", "bad", "good", "good"]
    unseen_instance = [0.33, 1]

    clf = MyKNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    y_pred = clf.predict([unseen_instance])

    assert y_pred[0] == "good"

    X_train = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    unseen_instance = [2, 3]

    clf = MyKNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    y_pred = clf.predict([unseen_instance])

    assert y_pred[0] == "yes"

    X_train = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]
    y_train = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",
            "-", "-", "+", "+", "+", "-", "+"]
    unseen_instance = [9.1, 11.0]

    clf = MyKNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict([unseen_instance])

    assert y_pred[0] == "+"

def test_dummy_classifier_fit():
    np.random.seed(0)
    X_train = [[val] for val in list(range(0, 100))]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    clf = MyDummyClassifier()
    clf.fit(X_train, y_train)

    assert clf.most_common_label == "yes"

    X_train = [[val] for val in list(range(0, 100))]
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    clf = MyDummyClassifier()
    clf.fit(X_train, y_train)

    assert clf.most_common_label == "no"

    X_train = [[val] for val in list(range(0, 100))]
    y_train = list(np.random.choice(["red", "blue", "green"], 100, replace=True, p=[0.1, 0.1, 0.8]))
    clf = MyDummyClassifier()
    clf.fit(X_train, y_train)

    assert clf.most_common_label == "green"

def test_dummy_classifier_predict():
    np.random.seed(0)
    X_train = [[val] for val in list(range(0, 100))]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    clf = MyDummyClassifier()
    clf.fit(X_train, y_train)

    X_test = [1, 2, 3, 4, 5]
    y_pred = clf.predict(X_test)

    assert y_pred == ["yes", "yes", "yes", "yes", "yes"]

    X_train = [[val] for val in list(range(0, 100))]
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    clf = MyDummyClassifier()
    clf.fit(X_train, y_train)

    X_test = [1, 2, 3, 4, 5]
    y_pred = clf.predict(X_test)

    assert y_pred == ["no", "no", "no", "no", "no"]

    X_train = [[val] for val in list(range(0, 100))]
    y_train = list(np.random.choice(["red", "blue", "green"], 100, replace=True, p=[0.1, 0.1, 0.8]))
    clf = MyDummyClassifier()
    clf.fit(X_train, y_train)

    X_test = [1, 2, 3, 4, 5]
    y_pred = clf.predict(X_test)

    assert y_pred == ["green", "green", "green", "green", "green"]

def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    clf = MyNaiveBayesClassifier()
    clf.fit(X_train_inclass_example, y_train_inclass_example)
    assert clf.priors == {"yes": 5 / 8, "no": 3 / 8}
    assert clf.posteriors == {"yes": [{1: 4 / 5, 2: 1 / 5}, {5: 2 / 5, 6: 3 / 5}], "no": [{1: 2 / 3, 2: 1 / 3}, {5: 2 / 3, 6: 1 / 3}]}

    # MA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    clf = MyNaiveBayesClassifier()
    clf.fit(X_train_iphone, y_train_iphone)
    assert clf.priors == {"no": 5 / 15, "yes": 10 / 15}
    assert clf.posteriors == {
        "no": [
            {1: 3 / 5, 2: 2 / 5},
            {3: 2 / 5, 1: 1 / 5, 2: 2 / 5},
            {"fair": 2 / 5, "excellent": 3 / 5}
        ],
        "yes": [
            {2: 8 / 10, 1: 2 / 10},
            {3: 3 / 10, 2: 4 / 10, 1: 3 / 10},
            {"fair": 7 / 10, "excellent": 3 / 10}
        ]
    }

    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    clf = MyNaiveBayesClassifier()
    clf.fit(X_train_train, y_train_train)
    assert clf.priors == {"on time": 14 / 20, "late": 2 / 20, "very late": 3 / 20, "cancelled": 1 / 20}
    assert clf.posteriors == {
        "on time": [
            {"weekday": 9 / 14, "saturday": 2 / 14, "holiday": 2 / 14, "sunday": 1 / 14},
            {"spring": 4 / 14, "winter": 2 / 14, "summer": 6 / 14, "autumn": 2 / 14},
            {"none": 5 / 14, "normal": 5 / 14, "high": 4 / 14},
            {"none": 5 / 14, "slight": 8 / 14, "heavy": 1 / 14}
        ],
        "late": [
            {"weekday": 1 / 2, "saturday": 1/2},
            {"winter": 2 / 2},
            {"high": 1 / 2, "normal": 1 / 2},
            {"heavy": 1 / 2, "none": 1 / 2}
        ],
        "very late": [
            {"weekday": 3 / 3},
            {"autumn": 1 / 3, "winter": 2 / 3},
            {"normal": 2 / 3, "high": 1 / 3},
            {"none": 1 / 3, "heavy": 2 / 3}
        ],
        "cancelled": [
            {"saturday": 1 / 1},
            {"spring": 1 / 1},
            {"high": 1 / 1},
            {"heavy": 1 / 1}
        ]
    }

def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    clf = MyNaiveBayesClassifier()
    clf.fit(X_train_inclass_example, y_train_inclass_example)

    X_test_inclass_example = [[1, 5]]
    y_pred_inclass_example = clf.predict(X_test_inclass_example)

    assert y_pred_inclass_example == ["yes"]

    # MA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    clf = MyNaiveBayesClassifier()
    clf.fit(X_train_iphone, y_train_iphone)

    X_test_iphone = [
        [2, 2, "fair"],
        [1, 1, "excellent"]
    ]
    y_pred_iphone = clf.predict(X_test_iphone)

    assert y_pred_iphone == ["yes", "no"]

    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    clf = MyNaiveBayesClassifier()
    clf.fit(X_train_train, y_train_train)

    X_test_train = [
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "high", "heavy"],
        ["sunday", "summer", "normal", "slight"]
    ]
    y_pred_train = clf.predict(X_test_train)

    assert y_pred_train == ["cancelled", "cancelled", "very late"]


def test_decision_tree_classifier_fit():
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
    interview_clf = MyDecisionTreeClassifier()
    interview_clf.fit(X_train_interview, y_train_interview)

    assert interview_clf.tree ==    ["Attribute", "att0",
                                        ["Value", "Junior",
                                            ["Attribute", "att3",
                                                ["Value", "no",
                                                    ["Leaf", "True", 3, 5]
                                                ],
                                                ["Value", "yes",
                                                    ["Leaf", "False", 2, 5]
                                                ]
                                            ]
                                        ],
                                        ["Value", "Mid",
                                            ["Leaf", "True", 4, 14]
                                        ],
                                        ["Value", "Senior",
                                            ["Attribute", "att2",
                                                ["Value", "no",
                                                    ["Leaf", "False", 3, 5] 
                                                ],
                                                ["Value", "yes",
                                                    ["Leaf", "True", 2, 5] 
                                                ]
                                            ]
                                        ]
                                    ]

    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    iphone_clf = MyDecisionTreeClassifier()
    iphone_clf.fit(X_train_iphone, y_train_iphone)

    print(iphone_clf.tree)

    assert iphone_clf.tree ==   ["Attribute", "att0",
                                    ["Value", 1, 
                                        ["Attribute", "att1",
                                            ["Value", 1,
                                                ["Leaf", "yes", 1, 5]
                                            ],
                                            ["Value", 2,
                                                ["Attribute", "att2",
                                                    ["Value", "excellent",
                                                        ["Leaf", "yes", 1, 2]
                                                    ],
                                                    ["Value", "fair",
                                                        ["Leaf", "no", 1, 2]
                                                    ]
                                                ]
                                            ],
                                            ["Value", 3,
                                                ["Leaf", "no", 2, 5]
                                            ]
                                        ]
                                    ],
                                    ["Value", 2, 
                                        ['Attribute', 'att2', 
                                            ['Value', 'excellent',
                                                ['Leaf', 'no', 2, 4]
                                            ], 
                                            ['Value', 'fair', 
                                                ['Leaf', 'yes', 6, 10]
                                            ]
                                        ]
                                    ]
                                ]

def test_decision_tree_classifier_predict():
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
    interview_clf = MyDecisionTreeClassifier()
    interview_clf.fit(X_train_interview, y_train_interview)

    X_test_interview = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    y_pred_interview = interview_clf.predict(X_test_interview)

    assert y_pred_interview == ["True", "False"]

    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    iphone_clf = MyDecisionTreeClassifier()
    iphone_clf.fit(X_train_iphone, y_train_iphone)

    X_test_iphone = [[2, 2, "fair"], [1, 1, "excellent"]]
    y_pred_iphone = iphone_clf.predict(X_test_iphone)

    assert y_pred_iphone == ["yes", "yes"]
