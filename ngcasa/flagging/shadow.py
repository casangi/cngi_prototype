# C
"""
this module will be included in the api
"""


def shadow(vis_dataset, **kwargs):  # args to be defined, storage_parms?):
    """
    .. todo::
        This function is not yet implemented

    Flag all baselines for antennas that are shadowed beyond the specified
    tolerance.

    All antennas in the zarr-file metadata (and their corresponding diameters)
    will be considered for shadow-flag calculations.
    For a given timestep, an antenna is flagged if any of its baselines
    (projected onto the uv-plane) is shorter than
       radius_1 + radius_2 - tolerance.
    The value of 'w' is used to determine which antenna is behind the other.
    The phase-reference center is used for antenna-pointing direction.

    Antennas that are not part of the observation, but still physically
    present and shadowing other antennas that are being used, must be added
    to the meta-data list in the zarr prior to calling this method.

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input dataset.
    TBD - shadowlimit or tolerance (in m)

    Returns:
    -------
    xds: xarray.core.dataset.Dataset
        Visibility dataset with updated flags
    """
    raise NotImplementedError('This method is not implemented')
