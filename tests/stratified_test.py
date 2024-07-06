import numpy as np
import torch

from subsample import stratified_subsample_multilabel


def test_stratified_subsample():
    y = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            5,
        ]
    )
    print("Unique in y with count:", np.unique(y, return_counts=True))
    print("len(y):", len(y))
    percentage = 0.5
    idxs = stratified_subsample_multilabel(y, percentage)
    print("Selected indexes:", idxs)
    # print the classes and their counts in selected indexes
    print(
        "Unique in selected indexes: with count:",
        np.unique(y[idxs], return_counts=True),
    )
    assert len(idxs) == int(len(y) * percentage)
    assert len(np.unique(y[idxs])) == len(np.unique(y))
    unq_classes, class_budget = np.unique(y, return_counts=True)
    for cl, budget in zip(unq_classes, class_budget):
        assert len(np.where(y[idxs] == cl)[0]) <= budget


def test_stratified_subsample_multilabel():
    y = [
        (),
        (0, 1),
        (0, 1),
        (0, 1),
        (1),
        (1),
        (1),
        (1),
        (1),
        (1),
        (1),
        (1),
        (2),
        (2, 3),
        (2, 3),
        (4),
        (1, 4),
        (1, 4),
        (1, 4),
        (4, 5),
        (4, 5),
        (4, 5),
        (4, 5),
    ]
    # print(np.unique(y, return_counts=True) )
    print("Y:", y)
    print("len(y):", len(y))
    percentage = 0.5
    idxs = stratified_subsample_multilabel(
        y, percentage, multilabel=True, classes=[0, 1, 2, 3, 4, 5]
    )
    print("Selected indexes:", idxs)
    # print the classes and their counts in selected indexes
    assert len(idxs) == int(len(y) * percentage)
    selected = [y[i] for i in idxs]
    print("Selected values:", selected)


def test_stratified_segmask():
    y0 = np.zeros(10)  # 0
    y1 = np.zeros(10)  # 0, 1
    y1[0:2] = 1
    y2 = np.ones(10)  # 1, 2
    y2[0:2] = 2
    y3 = np.full(10, 3)  # 2, 3
    y3[0:2] = 2
    y4 = np.full(10, 4)  # 1, 4
    y4[0:2] = 1
    y5 = np.full(10, 5)  # 5
    y6 = np.full(10, 5)  # 1, 5
    y6[0:2] = 1
    y7 = np.full(10, 5)  # 1, 5
    y7[0:2] = 1
    y8 = np.full(10, 5)  # 1, 5
    y8[0:2] = 1
    y9 = np.full(10, 5)  # 1, 5
    y9[0:2] = 1
    y10 = torch.full((10,), 5)  # 1, 5
    y10[0:2] = 1
    y11 = np.full((10,), 5)  # 1, 5
    y11[0:2] = 0
    y11[2:4] = 1
    y11[4:6] = 2
    y11[6:8] = 3
    y11[9] = 4
    y12 = y11.copy()

    y = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]  # , y11, y12]
    print("y:", y)

    percentage = 0.5
    idxs = stratified_subsample_multilabel(
        y, percentage, multilabel=True, classes=[0, 1, 2, 3, 4, 5]
    )
    print("Selected indexes:", idxs)
    assert len(idxs) == int(len(y) * percentage)
    selected = [y[i] for i in idxs]
    print("Selected values:", selected)
