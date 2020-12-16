#   Copyright 2019 AUI, Inc. Washington DC, USA
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
"""
this module will be included in the api
"""

#############################################
def describe_vis(infile):
    """
    Summarize the contents of a zarr format Visibility directory on disk

    Parameters
    ----------
    infile : str
        input filename of zarr Visibility data

    Returns
    -------
    pandas.core.frame.DataFrame
        Summary information
    """
    import os
    import numpy as np
    import pandas as pd
    from xarray import open_zarr
    
    infile = os.path.expanduser(infile)  # does nothing if $HOME is unknown
    summary = pd.DataFrame([])
    parts = os.listdir(infile)
    for ii, part in enumerate(parts):
        if part.startswith('global'): continue
        print('processing partition %i of %i' % (ii+1, len(parts)), end='\r')
        xds = open_zarr(os.path.join(infile, str(part)))
        sdf = {'xds': part, 'spw_id':xds.spw_id.values[0], 'pol_id':xds.pol_id.values[0],
               'times': len(xds.time),
               'baselines': len(xds.baseline),
               'chans': len(xds.chan),
               'pols': len(xds.pol),
               'size_MB': np.ceil(xds.nbytes / 1024 ** 2).astype(int)}
        summary = pd.concat([summary, pd.DataFrame(sdf, index=[ii])], axis=0, sort=False)
    
    print(' '*50, end='\r')
    return summary.set_index('xds').sort_index()
