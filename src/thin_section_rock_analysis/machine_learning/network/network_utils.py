from functools import reduce

import numpy as np
from sklearn.utils import class_weight


experiment1_data = {'target': None, 'predictions': [], 'files': []}


def get_class_weights(data):
    """
    Get class weights by collecting class frequencies
    Args:
        data: Tensor Dataset

    Returns:
        Dictionary with class weights
    """
    data_files = data.file_paths
    data_classes = []
    # Calculate class frequencies
    for class_label, class_name in enumerate(data.class_names):
        data_classes.extend(
            [class_label for file_name in data_files if file_name.find(class_name) >= 0])
    # Compute class weights based on frequency and save them in a dictionary
    weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(data_classes), y=data_classes)
    return {i: v for i, v in enumerate(weights)}


def reduce_to_group(data, groups):
    """
    Groups images based on sample ID and combines confidences to find the label to assign to that group
    Args:
        data: the data specifying the samples
        groups: all sample ids

    Returns:
        Summed confidences divided by total number of pictures in that group
    """
    d = {}
    for i, value in enumerate(data):
        group = groups[i]
        if group in d:
            d[group].append(value)
        else:
            d[group] = [value]

    return list(map(lambda l: reduce(lambda v, s: v + s, l) / len(l), d.values()))




# if __name__ == '__main__':
    # performance('output/binary-model-tl-ft-0.96.h5')
    # main()
