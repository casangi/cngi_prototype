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
        The flag variable name to remove

    Returns:
    -------
    xds: xarray.core.dataset.Dataset
        Visibility dataset without the removed flag variable
    """
    if name not in vis_dataset.variables:
        raise RuntimeError('Flag variable not found in dataset: {}'.
                           format(name))

    xds = vis_dataset.copy()
    flags_attr = _ensure_flags_attr(xds)
    del xds.attrs[flags_attr][name]

    return xds
