# C
"""
this module will be included in the api
"""
import xarray as xr


def manual_unflag(vis_dataset, selections):  # , storage_parms?):
    """
    Unflags the selected data. Flags corresponding to the selections are unset.

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input dataset.
    sel : list of dictionaries
        List of selections, each expressed as an xarray selection dictionary

    Returns:
    -------
    xds: xarray.core.dataset.Dataset
        Visibility dataset with updated (unset) flags
    """
    def unflag_with_reindex_like(vis_dataset, selections):
        flag_var = 'FLAG'
        xds = vis_dataset.assign()
        for sel in selections:
            fsel = xds[flag_var].sel(sel)
            if 0 in fsel.shape:
                print('WARNING: selection results in 0 shape. Sel: {}. Shape: {}'.
                      format(sel, fsel.shape))
            else:
                unflag_slice = xr.zeros_like(fsel, dtype=bool)
                reindexed_slice = unflag_slice.reindex_like(xds[flag_var], fill_value=True)
                xds[flag_var] &= reindexed_slice

        return xds

    if not isinstance(selections, list):
        raise ValueError('Parameter selection must be a list of selection dictionaries')
    return unflag_with_reindex_like(vis_dataset, selections)
