# C
"""
this module will be included in the api
"""
import numpy as np
import xarray as xr
import copy
# Use xdsio.vis_xds_packager to return an mxds?
# import cngi._helper.io as xdsio
from ._flagging_utils._antenna_to_baseline import _antenna_to_baseline


def manual_flag(mxds, xds_idx, commands=None, cmd_file=None):
    """
    Implements the 'manual' flagging method (equivalent to CASA6's flagdata manual
    mode).

    Parameters
    ----------
    mxds: xarray.core.dataset.Dataset
        Input Dataset
    xds_idx: int
        Index of the xarray datasets to get counts from (index in the xds'i'
        attributes of mxds). This is an oversimplification (early prototyping)
    commands: List[Dict]
        List of selection commands. Each item in the list represent one
        selection of data to flag. Mutually exclusive with cmd_file
        Every selection item is a dictionary. Selection is currently supported
        by 'time', 'chan', 'antenna', and 'pol'.
    cmd_file: filename
        Name of a file with text flagging commands, using the same format as
        used in the CASA6 pipelines for the "*flagonline.txt" or
        "*flagcmds.txt" files.
    TBD - Additional selection parameters / criteria

    Returns:
    -------
    xarray.core.dataset.Dataset
        Dataset with flags set on the selections given in either commands or
        cmd_file
    """
    xds = mxds.attrs['xds' + '{}'.format(xds_idx)]
    if commands and cmd_file:
        raise RuntimeError('Use either commands (to give a list of commands),'
                           'or cmd_file (to load the list of commands from a '
                           'file)')
    return _manual_with_reindex_like(mxds, xds, commands)


def _manual_with_reindex_like(mxds, xds, cmds):
    commands = copy.deepcopy(cmds)
    flag_var = 'FLAG'

    ret_xds = xds.assign()
    for cmd in commands:
        selection = _parse_sel_cmd(mxds, xds, cmd)
        fsel = xds[flag_var].sel(selection)
        if 0 in fsel.shape:
            print('WARNING: selection results in 0 shape. Sel: {}. Shape: {}'.
                  format(sel, fsel.shape))
            continue

        # Alternative 1 with .sel(): sel-ones, broadcast, or
        flag_slice = xr.ones_like(fsel, dtype=bool)
        reindexed_slice = flag_slice.reindex_like(xds[flag_var], fill_value=False)
        ret_xds[flag_var] |= reindexed_slice

    return ret_xds


def _parse_sel_cmd(mxds, xds, cmd):
    """
    Turn a 'manual' selection command into a dictionary that can be passed to
    xarray's sel(). This is just a basic implementation for demonstration purposes.

    Parameters
    ----------
    cmd: dict
        Selection dictionay as given to manual_flag
    mxds: xarray.core.dataset.Dataset
        Input dataset of datasets
    xds: xarray.core.dataset.Dataset
        Input dataset on which flagging will be applied
    Returns:
    -------
    dict
        Dictionary filtered and adapted to be used with sel() type of selections
    """
    result = {}
    # accept str end/stop and convert to datetime if needed
    if 'time' in cmd and isinstance(cmd['time'], slice):
        start, stop = cmd['time'].start, cmd['time'].stop
        if isinstance(start, str):
            start = np.datetime64(start)
        if isinstance(stop, str):
            stop = np.datetime64(stop)
        result['time'] = slice(start, stop)

    if 'antenna' in cmd:
        result['baseline'] = _antenna_to_baseline(mxds, xds, cmd['antenna'])

    if 'chan' in cmd:
        result['chan'] = chan

    if 'pol'in cmd:
        result['pol'] = pol
    print(' *** Going for cmd: {}, selection: {}'.format(cmd, result))
    return result
