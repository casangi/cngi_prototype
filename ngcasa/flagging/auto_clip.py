# C
"""
this module will be included in the api
"""


def auto_clip(vis_dataset, clip_min, clip_max):  # , storage_parms?):
    """
    Apply the clip flagging method. Data with values lower than clip_min
    or bigger than clip_max are flagged. Values are compared against the abs
    of the visibility values (no other expression supported at the moment).

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input dataset.
    clip_min : float
        Minimum below which data should be flagged
    max_clip : float
        Maximum above which data should be flagged

    Returns:
    -------
    xds: xarray.core.dataset.Dataset
        Visibility dataset with updated flags
    """
    flag_var = 'FLAG'
    data_var = 'DATA'

    to_clip = (abs(vis_dataset[data_var]) < clip_min) |\
              (abs(vis_dataset[data_var]) > clip_max)
    xds = vis_dataset.assign()
    xds[flag_var] = vis_dataset[flag_var] | to_clip

    # ? return _store(xds, list_xarray_data_variables, _storage_parms)
    return xds
