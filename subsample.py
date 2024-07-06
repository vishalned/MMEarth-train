import numpy as np
import torch
np.random.seed(1)
def remove_index(data, indexes_to_remove):
  """
  Removes a given index from a dictionary of lists.

  Args:
    data: A dictionary where the values are lists of indexes.
    indexes_to_remove: The index to remove.

  Returns:
    A new dictionary with the index removed from all lists.
  """
  ndata = {}
  for key, value in data.items():
      new_value = [i for i in value if i not in indexes_to_remove]
      ndata[key] = new_value
  return ndata


def stratified_subsample_multilabel(y, percentage=None, num_samples = None, multilabel=False, classes=[]):
    """
    Sub-sample a dataset to have a certain percentage of the original samples.
    The sub-sampling is done in a stratified way, i.e., the same proportion of samples from each class is kept.
    y: list or array-like, shape (n_samples,)
        The target variable
    percentage: float, default=0.1
        The percentage of samples to be kept
    multilabel: bool, default=False
        If True, y is a list of lists or arrays, where each element is a list or array of labels for each sample.
    classes: list, default=[]
        The classes to be considered. Required when multilabel is set to True.
    Returns:
         the indexes of the sub-sampled dataset
    """
    # we can either use percentage or num_samples
    if percentage is None:
        tot_samples = num_samples
    else:
        # assert 0 < percentage < 1, "Percentage must be between 0 and 1"
        tot_samples = int(percentage * len(y))
   
    assert (len(classes) > 0) or not multilabel, f"Classes must be provided, when multilabel({multilabel}) is True. This is to simply the code."
    if percentage == 1 or tot_samples == len(y):
        return y

    # Create cl_dict dictionary with the classes as key and the indexes where they are found as values
    if not multilabel:
        if not classes:
            classes = np.unique(y)
        cl_dict = {}
        for c in classes:
            cl_dict[c] = np.where(y == c)[0]
    else:
        cl_dict = {cl:[] for cl in classes}
        # Iterate the samples and add the index to the corresponding class(es)
        for i in range(len(y)):
            for cl in classes:
                # y[i] may be a tuple, array, torch.Tensor or a single value
                if isinstance(y[i], (set, list, tuple, np.ndarray, torch.Tensor)):
                    if cl in y[i]:
                        cl_dict[cl].append(i)
                elif isinstance(y[i], (int, str)):
                    if cl == y[i]:
                        cl_dict[cl].append(i)
                else:
                    raise ValueError(f"y[{i}] is not a valid type: {type(y[i])}")


    # Calculate the number of samples to be taken from each class, trying to keep the same proportion
    per_class = np.full(len(classes), tot_samples // len(classes))

    # If the number of samples is not enough, take the maximum possible
    per_class = np.minimum(per_class, [len(cl_dict[cl]) for cl in classes])

    # Iterate the per_class in increasing order of the number of samples
    per_class_sorted, classes_sorted = zip(*sorted(zip(per_class, classes), key=lambda x: x[0]))

    # Final indexes to return
    idxs = []

    # Iterate the classes in increasing order of the number of samples
    for cl, n in zip(classes_sorted, per_class_sorted):
        # The min is to avoid taking more samples than available; cl_dict is updated in the method itself
        nidx = np.random.choice(cl_dict[cl], size=min(n, len(cl_dict[cl])), replace=False)
        # Remove the selected indexes from the dictionary; An index may be in multiple classes
        cl_dict = remove_index(cl_dict, nidx)
        idxs.extend(nidx)

    # Get the rest in a round-robin fashion
    extra_samples = tot_samples - len(idxs)
    cl_idx = 0
    for i in range(extra_samples):
        budget_found = False
        while not budget_found:
            if len(cl_dict[cl_idx]) > 0:
                nidx = np.random.choice(cl_dict[cl_idx])
                idxs.append(nidx)
                # Remove from cl_dict
                cl_dict = remove_index(cl_dict, [nidx])
                budget_found = True
            # Move to the next class in either case
            cl_idx = (cl_idx + 1) % len(classes)

    return np.random.permutation(idxs)



