# C
"""
this module will be included in the api
"""


def auto_uvbin(vis_dataset, **kwargs):  # args to be defined, storage_parms?):
    """
    .. todo::
        This function is not yet implemented

    An autoflag algorithm that detects outliers on the gridded spatial frequency
    plane (Algorithm prototype exists).

    TBD: How can this method call  ngcasa.imaging._make_grid() and also satisfy
    code structure rules?

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input dataset.
    TBD

    Returns:
    -------
    xds: xarray.core.dataset.Dataset
        Visibility dataset with updated flags
    """
    raise NotImplementedError('This method is not implemented')
