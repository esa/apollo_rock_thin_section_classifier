import numpy as np
from sklearn.model_selection import StratifiedKFold
from itertools import chain


def flatten(l):
    """
    Returns flattened object on highest level of iterable collection
    Args:
        l - iterable of iterable collections e.g. [(1, 2), (3, )].
    Returns:
        iterable with the same type with the first layer flattened
    """
    if type(l) is np.ndarray:
        return np.array(list(chain(*l)))
    return type(l)(chain(*l))


def map_over_every_value(mapping_function, list_structure):
    """
    Applies function for every element of multidimensional structure which is not iterable
    Args:
        mapping_function - function to be applied on non-iterable element
        list_structure - input structure, may contain different iterable substructures
    Return value:
        structure with the same shape as list_structure with every non-iterable element modified
    """
    if hasattr(list_structure, '__iter__'):
        mapping = map(lambda l: map_over_every_value(mapping_function, l), list_structure)
        if type(list_structure) is np.ndarray: 
            return np.array(list(mapping), dtype=object)
        else:
            # pass a list to constructor
            return type(list_structure)(mapping)
    else:
        return mapping_function(list_structure)


def flatten_last_layer(l):
    """
    Given structure made up of iterables returns the structure where last layer is flattened.
    Example:
        ([(1,2), (3,4)], [(5,), (7, 8)]) -> ([1,2,3,4], [5,7,8])
    Args:
        l - iterable of iterable collections e.g. [(1, 2), (3, )].
    Returns:
        iterable where the last layer is flattened
    """  
    if not hasattr(l, '__iter__') or not hasattr(l[0], '__iter__'):
        return l  # just list or non iterable object

    if hasattr(l[0][0], '__iter__'):
        if type(l) is np.ndarray:
            return np.array(list(map(flatten_last_layer, l)), dtype=object)
        else:
            # pass list to constructor
            return type(l)(map(flatten_last_layer, l))
    else:
        return flatten(l)


class StratifiedGroupKFold:
    """
    Wraps stratified k-fold in order to preserve group position
    i.e. every sample in the group belongs to the same fold
    Interface similar as sklearn GroupKFold
    """
    def __init__(self, n_splits=3, shuffle=False):
        self.kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    
    def split(self, X, y, group):
        zipped = zip(X, y, group)

        d = dict()
        # Dictionary d maps group id to a structure ( [list of images], category )
        for x in zipped:
            if x[2] not in d:
                d[x[2]] = ([x[0]], x[1])
            else:
                d[x[2]][0].append(x[0]) 

        # Explode values into X = [list of images], y = category
        X, y = zip(*d.values())

        # Perform k-fold validation
        folds = list(self.kfold.split(X, y))

        # X is in the form of list of lists of images. we want to transition into a flat list.
        # To fix indexes in folds we need map from index of sample into index of image
        # creates indices = maps index -> list of indexes in exploded array
        indices = []
        i = 0
        for x in X:
            sub_list = []
            for _ in range(len(x)):
                sub_list.append(i)
                i += 1
            indices.append(sub_list)
        # Flatten the X array
        X = list(flatten(X))
        # flatten the Y array and transform it to have it be the same shape as X
        y = [[a]*b for a, b in zip(y, [len(t) for t in indices])]
        y = flatten(y)

        # Map folds from sample index into image index
        # replaces index of sample with list of index of images
        folds = map_over_every_value(lambda l: indices[l], folds)
        # Flattens last layer of structure - so that lists contains list of image indices
        folds = flatten_last_layer(folds)
 
        return folds, X, y
