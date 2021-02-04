# C
"""
this module will be included in the api
"""
import xarray as xr
from ._flagging_utils._flag_attributes import _add_descr


def manager_add(vis_dataset, name, descr, source=None):
    """
    Add a new flag variable to the dataset. All flags in the new variable take
    the values from the source flag variable. If no source is found, the flags
    are all set to false.

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input dataset
    name : string
        The new flag variable will be named FLAG_name
    descr : string
        Text description of the flag variable (for example, 'backup_beginning')
    source : name
        Name of an existing flag variable. If specified its values will be used
        to initialize the new flag variable being added

    Returns:
    -------
    xds: xarray.core.dataset.Dataset
        Visibility dataset with updated set of flag variables
    """
    flag_var = 'FLAG'
    add_name = '{}_{}'.format(flag_var, name)
    if add_name in vis_dataset.variables:
        raise RuntimeError('Flag variable already in dataset: {}'.
                           format(add_name))

    # Add variable
    source_name = '{}_{}'.format(flag_var, source)
    if source_name in vis_dataset.variables:
        if source_name not in vis_dataset.variables:
            raise RuntimeError('Source variable not found in dataset: {}'.
                               format(source_name))
        xds = vis_dataset.copy()
        xds[add_name] = vis_dataset[source_name]
    else:
        xds = vis_dataset.copy()
        xds[add_name] = xr.zeros_like(vis_dataset[flag_var], dtype=bool)

    _add_descr(xds, add_name, descr)

    return xds
