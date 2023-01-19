def check_key(element, *keys):
    """
    Check if *keys (nested) exists in dict
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
