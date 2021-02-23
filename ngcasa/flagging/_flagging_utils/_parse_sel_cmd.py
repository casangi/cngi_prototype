# C
import numpy as np
from ._antenna_to_baseline import _antenna_to_baseline

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
        result['chan'] = cmd['chan']

    if 'pol'in cmd:
        result['pol'] = cmd['pol']

    return result
