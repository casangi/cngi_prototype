# C

def _ensure_flags_attr(xds):
    """ Returns the name of the dataset attribute that holds info on flag variables. """
    flag_var = 'FLAG'
    flags_attr = 'flag_variables'
    if flags_attr not in xds.attrs:
        xds.attrs[flags_attr] = {flag_var: 'Default flags variable'}
    return flags_attr

def _add_descr(xds, add_name=None, descr=None):
    """ Adds into the dict of flag variables a new flag variable and its description. """
    flags_attr = _ensure_flags_attr(xds)
    if add_name:
        xds.attrs[flags_attr][add_name] = descr
