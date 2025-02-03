# New Datasets

To use the finetuning scripts for new datasets, you need to change some parts of the code based on your new dataset. This readme describes the various components to change.

- Update the parser argument `--data_set` choices with your new dataset name (make sure this dataset name is used in the other parts of the code too).
- Add loss: Add a new loss for your dataset name in the `criterion_fn()` in `main_finetune.py`.
- Dataloaders: You can define a custom dataloader for your dataset in a new file and call it in the `main_finetune.py` file in line 465. The dataloader returns the 2 arguments. The first is a tuple containing the train, val, test dataloaders and the second is a task dictionary that has the following keys:
    - class_names: describing the various class names if it is a classification problem, else None.
    - num_classes: number of classes
    - type: classification or segmentation (if you need a regression task, you would have to make some changes in the code)
    - label_type: classification, multi_label_classification, segmentation (what kind of labels you have)
- Metric definition: Define what metric you want to use in `eval_metrics_generator()` in `engine_finetune.py`.


