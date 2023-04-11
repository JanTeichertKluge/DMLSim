def check_key(element, *keys):
    """
    Check if a dictionary `element` contains a nested dictionary key path `keys`.

    Args:
    -------
        element (dict): The dictionary to check for the key path.
        *keys (tuple): A variable number of string arguments that represent the nested dictionary key path.

    Returns:
    -------
        bool: True if the key path exists in the dictionary, False otherwise.

    Raises:
    -------
        AttributeError: If `element` is not a dictionary or if fewer than two arguments are passed in.

    Example:
    -------
        >>> d = {'a': {'b': {'c': 'd'}}}
        >>> check_key(d, 'a', 'b', 'c')
        True
        >>> check_key(d, 'a', 'b', 'd')
        False
    """
    if not isinstance(element, dict):
        raise AttributeError("Expects dict as first argument.")
    if len(keys) == 0:
        raise AttributeError("check_key expects at least two arguments, one given.")

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True
