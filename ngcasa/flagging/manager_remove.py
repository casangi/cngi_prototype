# C
"""
this module will be included in the api
"""
from ._flagging_utils._flag_attributes import _ensure_flags_attr


def manager_remove(vis_dataset, name):
    """
    Remove flag variable from the dataset.

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input dataset
    name : string
        The flag variable name to remove (FLAG_name)

    Returns:
    -------
    xds: xarray.core.dataset.Dataset
        Visibility dataset without the removed flag variable
    """
    flag_var = 'FLAG'
    rem_name = '{}_{}'.format(flag_var, name)
    if rem_name not in vis_dataset.variables:
        raise RuntimeError('Flag variable not found in dataset: {}'.
                           format(rem_name))

    xds = vis_dataset.copy()
    flags_attr = _ensure_flags_attr(xds)
    del xds.attrs[flags_attr][rem_name]

    return xds
