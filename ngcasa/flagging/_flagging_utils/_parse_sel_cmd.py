#   Copyright 2020-21 European Southern Observatory, ALMA partnership
#   Copyright 2020 AUI, Inc. Washington DC, USA
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
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
