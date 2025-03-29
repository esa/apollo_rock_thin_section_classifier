import pickle


def load_file(file):
    """
    Load pickle file
    Args:
        file: which file to load

    Returns:
        unpickled file
    """
    with open(file, "rb") as f:
        return pickle.load(f)


def write_to_file(data, file):
    """
    Write to pickle file
    Args:
        data: data to write to file specified
        file: name of destination file

    Returns:
        Does not return, but saves file
    """
    with open(file, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def get_filename(path):
    """
    Finds the latest occurrence of '/' and thus only the filename without the rest of the path
    Args:
        path: full path of the file

    Returns:
        only the file name in lowercase
    """
    return path[path.rfind("/")+1:].lower()


def join_dicts(dict1, dict2):
    """
    Helps to merge two dictionaries. Also takes into account that dictionaries may share keys
    and should be appended to each other rather than overwritten.
    You can find an explanation here: https://datagy.io/python-merge-dictionaries/ why you should do it this way.
    Args:
        dict1: dictionary to join with dict2 dictionary
        dict2: dictionary to join with dict1 dictionary

    Returns:
        Joined dictionary
    """
    # build a list of all the keys
    all_keys = list(dict1.keys())
    all_keys.extend(list(dict2.keys()))
    all_keys = list(set(all_keys))

    new_dict = {}
    for key in all_keys:
        t = {}
        if key in dict1:
            v = dict1[key]
            # the ** operator is an unpacking operator used to access both the key and value
            # It involves adding every item from multiple dictionaries to a new dictionary
            t = {**t, **v}
        if key in dict2:
            v = dict2[key]
            t = {**t, **v}
        new_dict[key] = t
    return new_dict
