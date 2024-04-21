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


if __name__ == "__main__":
    def test_stratified_subsample():
        y = np.array([ 0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,3,3,3,3,4,4,4,4,4,4,4,5])
        print("Unique in y with count:", np.unique(y, return_counts=True) )
        print("len(y):", len(y))
        percentage = 0.5
        idxs = stratified_subsample_multilabel(y, percentage)
        print("Selected indexes:", idxs)
        # print the classes and their counts in selected indexes
        print("Unique in selected indexes: with count:",np.unique(y[idxs], return_counts=True))
        assert len(idxs) == int(len(y) * percentage)
        assert len(np.unique(y[idxs])) == len(np.unique(y))
        unq_classes, class_budget = np.unique(y, return_counts=True)
        for cl, budget in zip(unq_classes, class_budget):
            assert len(np.where(y[idxs] == cl)[0]) <= budget

    def test_stratified_subsample_multilabel():
        y = [(), (0,1),(0,1),(0,1),(1),(1),(1),(1),(1),(1),(1),(1),(2),(2,3),(2,3),(4),(1,4),(1,4),(1,4),(4,5),(4,5),(4,5),(4,5)]
        # print(np.unique(y, return_counts=True) )
        print("Y:",y)
        print("len(y):",len(y))
        percentage = 0.5
        idxs = stratified_subsample_multilabel(y, percentage, multilabel=True, classes=[0,1,2,3,4,5])
        print("Selected indexes:", idxs)
        # print the classes and their counts in selected indexes
        assert len(idxs) == int(len(y) * percentage)
        selected = [y[i] for i in idxs]
        print("Selected values:", selected)

    def test_stratified_segmask():
        y0 = np.zeros(10) # 0
        y1 = np.zeros(10) # 0, 1
        y1[0:2] = 1
        y2 = np.ones(10) # 1, 2
        y2[0:2] = 2
        y3 = np.full(10, 3) # 2, 3
        y3[0:2] = 2
        y4 = np.full(10, 4) # 1, 4
        y4[0:2] = 1
        y5 = np.full(10,5) # 5
        y6 = np.full(10,5) # 1, 5
        y6[0:2] = 1
        y7 = np.full(10,5) # 1, 5
        y7[0:2] = 1
        y8 = np.full(10,5) # 1, 5
        y8[0:2] = 1
        y9 = np.full(10,5) # 1, 5
        y9[0:2] = 1
        y10 = torch.full((10,),5) # 1, 5
        y10[0:2] = 1
        y11 = np.full((10,),5) # 1, 5
        y11[0:2] = 0
        y11[2:4] = 1
        y11[4:6] = 2
        y11[6:8] = 3
        y11[9] = 4
        y12 = y11.copy()


        y = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]#, y11, y12]
        print("y:", y)

        percentage = 0.5
        idxs = stratified_subsample_multilabel(y, percentage, multilabel=True, classes=[0, 1, 2, 3, 4, 5])
        print("Selected indexes:", idxs)
        assert len(idxs) == int(len(y) * percentage)
        selected = [y[i] for i in idxs]
        print("Selected values:", selected)

    test_stratified_subsample()
    print("test_stratified_subsample passed")
    test_stratified_subsample_multilabel()
    print("test_stratified_subsample_multilabel passed")
    test_stratified_segmask()
    print("test_stratified_segmask passed")
