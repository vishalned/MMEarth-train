import pytest
import torch
from sklearn.metrics import f1_score
from torchmetrics.classification import MultilabelF1Score


def test_f1_score_alignment_multilabel():
    # Define true and predicted labels for a multilabel classification problem
    y_true = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1]
    ]
    y_pred = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 1]
    ]

    # Calculate F1 scores with scikit-learn
    expected_f1_macro = f1_score(y_true, y_pred, average='macro')
    expected_f1_micro = f1_score(y_true, y_pred, average='micro')

    # Convert lists to tensors
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    # Calculate F1 scores with torchmetrics
    f1_macro = MultilabelF1Score(num_labels=3, average='macro')
    f1_micro = MultilabelF1Score(num_labels=3, average='micro')

    for i in range(5):
        [f1_macro(y_pred_tensor[[i]], y_true_tensor[[i]]) for i in range(len(y_true))]
        f1_score_macro_value = f1_macro.compute()
        [f1_micro(y_pred_tensor[[i]], y_true_tensor[[i]]) for i in range(len(y_true))]
        f1_score_micro_value = f1_micro.compute()

    # Assertions to check alignment
    assert pytest.approx(f1_score_macro_value.item(), 0.001) == expected_f1_macro
    assert pytest.approx(f1_score_micro_value.item(), 0.001) == expected_f1_micro


if __name__ == '__main__':
    pytest.main()
