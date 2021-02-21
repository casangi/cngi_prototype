# C
"""
this module will be included in the api
"""
import numpy as np
import xarray as xr
import copy


def manual_flag(mxds, xds_idx, commands):
    """
    Implements the 'manual' flagging method (equivalent to CASA6's flagdata manual
    mode).
    ... 

    Note: per-SPW counts are not included in this prototype version. Handling
    of multi-SPWs operations is subject to changes depending on interface
    design at (or around) the ngCASA layer.

    Parameters
    ----------
    mxds: xarray.core.dataset.Dataset
        Input Dataset
    xds_idx: int
        TBD - describe
    commands: List[Dict]
        TBD - describe
        TBD - describe
    TBD - Additional selection parameters / criteria

    Returns:
    -------
    xarray.core.dataset.Dataset
        Dataset with flags... 
        Describe - TBD
    """
    xds = mxds.attrs['xds' + '{}'.format(xds_idx)]
    return manual_with_reindex_like(xds, commands)


def manual_with_reindex_like(xds, cmds):
    commands = copy.deepcopy(cmds)

    for cmd in commands:
        if 'antenna' in cmd:
            cmd['baseline'] = cmd['antenna']
            cmd.pop('antenna', None)

        if 'time' in cmd:
            start, stop = cmd['time'].start, cmd['time'].stop
            if isinstance(start, str):
                start = np.datetime64(start)
            if isinstance(stop, str):
                stop = np.datetime64(stop)
            cmd['time'] = slice(start, stop)

        fsel = xds.FLAG.sel(cmd)
        if 0 not in fsel.shape:
            flag_slice = xr.ones_like(fsel, dtype=bool)
            reindexed_slice = flag_slice.reindex_like(xds.FLAG, fill_value=False)
            xds['FLAG'] |= reindexed_slice

    return xds
